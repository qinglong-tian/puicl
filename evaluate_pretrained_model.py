from __future__ import annotations

import argparse
import importlib
import json
import sys
import zipfile
from io import BytesIO, TextIOWrapper
from pathlib import Path
from typing import Dict, Optional
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from scipy.io import arff
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)


UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases"

METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "average_precision",
    "fpr_at_tpr_0_80",
    "fpr_at_tpr_0_90",
    "fpr_at_tpr_0_95",
    "outlier_score_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate pretrained_model/latest.pt using an embedded benchmark protocol: "
            "fixed datasets, fixed PU conversion, and fixed metrics."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--allow-uci-download", action="store_true", default=True)
    parser.add_argument("--no-uci-download", dest="allow_uci_download", action="store_false")
    parser.add_argument("--n-replicates", type=int, default=10)
    parser.add_argument("--max-attempts-per-dataset", type=int, default=200)
    parser.add_argument("--max-positive-size", type=int, default=300)
    parser.add_argument("--unlabeled-positive-ratio", type=int, default=1)
    parser.add_argument("--labeled-positive-ratio", type=int, default=4)
    parser.add_argument("--outlier-rate", type=float, default=5.0 / 6.0)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--max-categorical-classes", type=int, default=64)
    return parser.parse_args()


def resolve_runtime_paths() -> tuple[Path, Path, str]:
    pretrain_root = Path(__file__).resolve().parent
    if not (pretrain_root / "__init__.py").exists() or not (pretrain_root / "model.py").exists():
        raise RuntimeError(f"Expected evaluator to live inside the package root, got: {pretrain_root}")

    repo_root = pretrain_root.parent
    package_name = pretrain_root.name
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root, pretrain_root, package_name


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        return "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if device_arg == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available() or not torch.backends.mps.is_built():
            raise RuntimeError("MPS requested but not available.")
    return device_arg


def resolve_cache_dir(requested: Optional[Path], pretrain_root: Path) -> Path:
    if requested is not None:
        return requested.expanduser()
    return pretrain_root / ".cache"


def _public_path_label(path: Path, repo_root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        if resolved.name:
            return f"<external>/{resolved.name}"
        return "<external>"


def load_model(checkpoint_path: Path, model_cls, device: str):
    payload = torch.load(checkpoint_path, map_location=device)
    model_cfg = payload.get("config", {}).get("model", {})
    model = model_cls(
        embedding_size=int(model_cfg.get("embedding_size", 128)),
        num_attention_heads=int(model_cfg.get("num_attention_heads", 8)),
        mlp_hidden_size=int(model_cfg.get("mlp_hidden_size", 256)),
        num_layers=int(model_cfg.get("num_layers", 6)),
        num_outputs=int(model_cfg.get("num_outputs", 2)),
    ).to(device)

    state_dict = payload.get("model_state_dict", payload)
    load_result = model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, payload, load_result


def _download_with_cache(
    url: str,
    subdir: str,
    filename: str,
    cache_dir: Path,
    allow_download: bool = True,
) -> Path:
    target_dir = cache_dir / "uci" / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    local_path = target_dir / filename

    if local_path.exists():
        return local_path
    if not allow_download:
        raise FileNotFoundError(
            f"UCI cached file not found and downloads are disabled: {local_path}. "
            "Set --allow-uci-download to fetch it."
        )

    urlretrieve(url, local_path)
    return local_path


def _strip_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    object_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        df[col] = df[col].astype("string").str.strip()
    return df


def _read_uci_table_from_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [
            name
            for name in zf.namelist()
            if not name.endswith("/") and not name.lower().startswith("__macosx/")
        ]
        if len(members) == 0:
            raise FileNotFoundError(f"No data files found in zip: {zip_path}")

        preferred_exts = (".csv", ".arff", ".data", ".txt", ".xlsx", ".xls")
        ordered_members: list[str] = []
        for ext in preferred_exts:
            ordered_members.extend([name for name in members if name.lower().endswith(ext)])
        ordered_members.extend([name for name in members if name not in ordered_members])

        last_error = None
        for selected in ordered_members:
            try:
                with zf.open(selected) as f:
                    lower = selected.lower()
                    if lower.endswith(".csv"):
                        df = pd.read_csv(f)
                    elif lower.endswith(".arff"):
                        with TextIOWrapper(f, encoding="utf-8", errors="ignore") as txt_f:
                            data, _ = arff.loadarff(txt_f)
                        df = pd.DataFrame(data)
                        for col in df.columns:
                            if df[col].dtype == object:
                                df[col] = df[col].apply(
                                    lambda value: value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value
                                )
                    elif lower.endswith(".xlsx") or lower.endswith(".xls"):
                        df = pd.read_excel(BytesIO(f.read()))
                    else:
                        df = pd.read_csv(f)
                return _strip_object_columns(df)
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"Failed to parse any file from zip: {zip_path}") from last_error


def get_benchmark_datasets(
    cache_dir: Path,
    allow_uci_download: bool = True,
    binary_seed: int = 42,
) -> list[Dict]:
    datasets = []
    root_missing_drop_threshold = 0.20

    wdbc_feature_names = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
    ]
    wdbc_cols = ["id", "target"] + wdbc_feature_names
    wdbc_path = _download_with_cache(
        f"{UCI_BASE_URL}/breast-cancer-wisconsin/wdbc.data",
        subdir="wdbc",
        filename="wdbc.data",
        cache_dir=cache_dir,
        allow_download=allow_uci_download,
    )
    wdbc_df = pd.read_csv(wdbc_path, header=None, names=wdbc_cols)
    datasets.append(
        {
            "name": "uci_wdbc_continuous",
            "source": "uci:wdbc",
            "X": wdbc_df[wdbc_feature_names].copy(),
            "y": wdbc_df["target"].copy(),
            "schema_hint": {"force_all_continuous": True},
        }
    )

    adult_cols = [
        "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
        "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
        "hours_per_week", "native_country", "target",
    ]
    adult_path = _download_with_cache(
        f"{UCI_BASE_URL}/adult/adult.data",
        subdir="adult",
        filename="adult.data",
        cache_dir=cache_dir,
        allow_download=allow_uci_download,
    )
    adult_df = pd.read_csv(adult_path, header=None, names=adult_cols, na_values=["?"], skipinitialspace=True)
    adult_df = _strip_object_columns(adult_df)
    adult_df = adult_df.dropna(axis=0).reset_index(drop=True)

    adult_categorical_cols = [
        "workclass", "education", "marital_status", "occupation", "relationship",
        "race", "sex", "native_country",
    ]
    adult_feature_cols = [col for col in adult_cols if col != "target"]
    adult_continuous_cols = [col for col in adult_feature_cols if col not in adult_categorical_cols]
    datasets.append(
        {
            "name": "uci_adult_mixed",
            "source": "uci:adult",
            "X": adult_df[adult_feature_cols].copy(),
            "y": adult_df["target"].copy(),
            "schema_hint": {
                "force_categorical_cols": adult_categorical_cols,
                "force_continuous_cols": adult_continuous_cols,
            },
        }
    )

    spambase_cols = [f"f{i}" for i in range(1, 58)] + ["target"]
    spambase_path = _download_with_cache(
        f"{UCI_BASE_URL}/spambase/spambase.data",
        subdir="spambase",
        filename="spambase.data",
        cache_dir=cache_dir,
        allow_download=allow_uci_download,
    )
    spambase_df = pd.read_csv(spambase_path, header=None, names=spambase_cols)
    datasets.append(
        {
            "name": "uci_spambase_continuous",
            "source": "uci:spambase",
            "X": spambase_df[[f"f{i}" for i in range(1, 58)]].copy(),
            "y": spambase_df["target"].copy(),
            "schema_hint": {"force_all_continuous": True},
        }
    )

    mushroom_feature_cols = [
        "cap_shape", "cap_surface", "cap_color", "bruises", "odor",
        "gill_attachment", "gill_spacing", "gill_size", "gill_color", "stalk_shape",
        "stalk_root", "stalk_surface_above_ring", "stalk_surface_below_ring",
        "stalk_color_above_ring", "stalk_color_below_ring", "veil_type", "veil_color",
        "ring_number", "ring_type", "spore_print_color", "population", "habitat",
    ]
    mushroom_cols = ["target"] + mushroom_feature_cols
    mushroom_path = _download_with_cache(
        f"{UCI_BASE_URL}/mushroom/agaricus-lepiota.data",
        subdir="mushroom",
        filename="agaricus-lepiota.data",
        cache_dir=cache_dir,
        allow_download=allow_uci_download,
    )
    mushroom_df = pd.read_csv(mushroom_path, header=None, names=mushroom_cols, na_values=["?"], skipinitialspace=True)
    mushroom_df = _strip_object_columns(mushroom_df)

    stalk_root_missing_rate = float(mushroom_df["stalk_root"].isna().mean())
    if stalk_root_missing_rate > root_missing_drop_threshold:
        mushroom_df = mushroom_df.drop(columns=["stalk_root"])

    mushroom_df = mushroom_df.dropna(axis=0).reset_index(drop=True)
    mushroom_X_cols = [col for col in mushroom_df.columns if col != "target"]
    datasets.append(
        {
            "name": "uci_mushroom_categorical",
            "source": "uci:mushroom",
            "X": mushroom_df[mushroom_X_cols].copy(),
            "y": mushroom_df["target"].copy(),
            "schema_hint": {"force_all_categorical": True},
        }
    )

    magic_feature_cols = [
        "fLength", "fWidth", "fSize", "fConc", "fConc1",
        "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist",
    ]
    magic_cols = magic_feature_cols + ["target"]
    magic_path = _download_with_cache(
        f"{UCI_BASE_URL}/magic/magic04.data",
        subdir="magic_gamma_telescope",
        filename="magic04.data",
        cache_dir=cache_dir,
        allow_download=allow_uci_download,
    )
    magic_df = pd.read_csv(magic_path, header=None, names=magic_cols)
    datasets.append(
        {
            "name": "uci_magic_gamma_continuous",
            "source": "uci:magic-gamma-telescope",
            "X": magic_df[magic_feature_cols].copy(),
            "y": magic_df["target"].copy(),
            "schema_hint": {"force_all_continuous": True},
        }
    )

    car_cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "target"]
    car_path = _download_with_cache(
        f"{UCI_BASE_URL}/car/car.data",
        subdir="car_evaluation",
        filename="car.data",
        cache_dir=cache_dir,
        allow_download=allow_uci_download,
    )
    car_df = pd.read_csv(car_path, header=None, names=car_cols)
    car_df = _strip_object_columns(car_df)

    car_binary_target = car_df["target"].map(
        {
            "unacc": "unacc_or_acc",
            "acc": "unacc_or_acc",
            "good": "good_or_vgood",
            "vgood": "good_or_vgood",
        }
    )
    car_valid = car_binary_target.notna()
    car_df = car_df.loc[car_valid].reset_index(drop=True)
    car_binary_target = car_binary_target.loc[car_valid].reset_index(drop=True)
    datasets.append(
        {
            "name": "uci_car_evaluation_categorical",
            "source": "uci:car-evaluation",
            "X": car_df[["buying", "maint", "doors", "persons", "lug_boot", "safety"]].copy(),
            "y": car_binary_target,
            "schema_hint": {"force_all_categorical": True},
        }
    )

    banknote_cols = ["variance", "skewness", "curtosis", "entropy", "target"]
    banknote_zip_path = _download_with_cache(
        "https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip",
        subdir="banknote_authentication",
        filename="banknote+authentication.zip",
        cache_dir=cache_dir,
        allow_download=allow_uci_download,
    )
    banknote_df = _read_uci_table_from_zip(banknote_zip_path)
    banknote_df.columns = banknote_cols
    datasets.append(
        {
            "name": "uci_banknote_authentication_continuous",
            "source": "uci:banknote-authentication",
            "X": banknote_df[["variance", "skewness", "curtosis", "entropy"]].copy(),
            "y": banknote_df["target"].copy(),
            "schema_hint": {"force_all_continuous": True},
        }
    )

    rice_zip_path = _download_with_cache(
        "https://archive.ics.uci.edu/static/public/545/rice+cammeo+and+osmancik.zip",
        subdir="rice_cammeo_osmancik",
        filename="rice+cammeo+and+osmancik.zip",
        cache_dir=cache_dir,
        allow_download=allow_uci_download,
    )
    rice_df = _read_uci_table_from_zip(rice_zip_path)
    rice_target_col = next(
        (candidate for candidate in ["Class", "class", "target", "Target"] if candidate in rice_df.columns),
        rice_df.columns[-1],
    )
    rice_feature_cols = [col for col in rice_df.columns if col != rice_target_col]
    datasets.append(
        {
            "name": "uci_rice_cammeo_osmancik_continuous",
            "source": "uci:rice-cammeo-and-osmancik",
            "X": rice_df[rice_feature_cols].copy(),
            "y": rice_df[rice_target_col].copy(),
            "schema_hint": {"force_all_continuous": True},
        }
    )

    default_credit_zip_path = _download_with_cache(
        "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
        subdir="default_credit_card_clients",
        filename="default+of+credit+card+clients.zip",
        cache_dir=cache_dir,
        allow_download=allow_uci_download,
    )
    default_credit_df = _read_uci_table_from_zip(default_credit_zip_path)
    default_credit_df = _strip_object_columns(default_credit_df)

    def _normalize_col_name(col_name: object) -> str:
        text = str(col_name).strip().lower()
        text = "".join(ch if ch.isalnum() else "_" for ch in text)
        while "__" in text:
            text = text.replace("__", "_")
        return text.strip("_")

    def _is_placeholder_col_name(col_name: object) -> bool:
        norm = _normalize_col_name(col_name)
        return norm.startswith("unnamed") or norm.isdigit()

    if default_credit_df.shape[0] > 0:
        first_row_norm = [_normalize_col_name(v) for v in default_credit_df.iloc[0].tolist()]
        header_keywords = {"id", "y", "limit_bal", "pay_0", "bill_amt1", "default_payment_next_month"}
        placeholder_cols = all(_is_placeholder_col_name(col) for col in default_credit_df.columns)
        if placeholder_cols or len(header_keywords & set(first_row_norm)) >= 2:
            default_credit_df.columns = default_credit_df.iloc[0].astype(str).tolist()
            default_credit_df = default_credit_df.iloc[1:].reset_index(drop=True)
            default_credit_df = _strip_object_columns(default_credit_df)

    if default_credit_df.shape[0] > 0:
        first_row = default_credit_df.iloc[0].astype("string").str.lower().tolist()
        col_tokens = [str(col).strip().lower() for col in default_credit_df.columns.tolist()]
        if sum(int(a == b) for a, b in zip(first_row, col_tokens)) >= max(3, len(col_tokens) // 2):
            default_credit_df = default_credit_df.iloc[1:].reset_index(drop=True)

    normalized_to_original = {_normalize_col_name(col): col for col in default_credit_df.columns}
    preferred_norm_names = [
        "y",
        "default_payment_next_month",
        "defaultpaymentnextmonth",
        "target",
    ]
    default_credit_target_col = next(
        (normalized_to_original[name] for name in preferred_norm_names if name in normalized_to_original),
        None,
    )

    if default_credit_target_col is None:
        contains_default_cols = [
            col
            for col in default_credit_df.columns
            if "default" in _normalize_col_name(col) and "month" in _normalize_col_name(col)
        ]
        if len(contains_default_cols) > 0:
            default_credit_target_col = contains_default_cols[0]

    if default_credit_target_col is None:
        binary_cols = [
            col
            for col in default_credit_df.columns
            if int(pd.Series(default_credit_df[col]).nunique(dropna=True)) == 2
        ]
        named_binary_cols = [
            col
            for col in binary_cols
            if _normalize_col_name(col) in {"y", "target"} or "default" in _normalize_col_name(col)
        ]
        if len(named_binary_cols) > 0:
            default_credit_target_col = named_binary_cols[0]
        elif len(binary_cols) > 0:
            default_credit_target_col = binary_cols[-1]

    if default_credit_target_col is None:
        raise ValueError("Could not infer target column for default credit dataset.")

    default_credit_target = pd.to_numeric(default_credit_df[default_credit_target_col], errors="coerce")
    valid_target = default_credit_target.notna()
    default_credit_df = default_credit_df.loc[valid_target].reset_index(drop=True)
    default_credit_target = default_credit_target.loc[valid_target].reset_index(drop=True)

    if int(default_credit_target.nunique(dropna=True)) != 2:
        unique_counts = {
            str(col): int(pd.Series(default_credit_df[col]).nunique(dropna=True))
            for col in default_credit_df.columns
        }
        raise ValueError(
            "Default credit target is not binary after parsing. "
            f"Selected target column={default_credit_target_col!r}, "
            f"num_classes={int(default_credit_target.nunique(dropna=True))}, "
            f"column_nunique={unique_counts}"
        )

    default_credit_feature_cols = [
        col
        for col in default_credit_df.columns
        if col != default_credit_target_col and _normalize_col_name(col) != "id"
    ]
    datasets.append(
        {
            "name": "uci_default_credit_card_clients_continuous",
            "source": "uci:default-of-credit-card-clients",
            "X": default_credit_df[default_credit_feature_cols].copy(),
            "y": default_credit_target.astype(np.int64).copy(),
            "schema_hint": {"force_all_continuous": True},
        }
    )

    abalone_cols = [
        "Sex", "Length", "Diameter", "Height", "Whole_weight", "Shucked_weight",
        "Viscera_weight", "Shell_weight", "Rings",
    ]
    abalone_path = _download_with_cache(
        f"{UCI_BASE_URL}/abalone/abalone.data",
        subdir="abalone",
        filename="abalone.data",
        cache_dir=cache_dir,
        allow_download=allow_uci_download,
    )
    abalone_df = pd.read_csv(abalone_path, header=None, names=abalone_cols)
    abalone_df = _strip_object_columns(abalone_df)

    rings_series = pd.to_numeric(abalone_df["Rings"], errors="coerce")
    rings_cutoff = int(np.nanmedian(rings_series.to_numpy()))
    abalone_binary_target = np.where(rings_series >= rings_cutoff, f"rings_ge_{rings_cutoff}", f"rings_lt_{rings_cutoff}")
    datasets.append(
        {
            "name": "uci_abalone_binary_rings_cutoff",
            "source": "uci:abalone",
            "X": abalone_df[[c for c in abalone_cols if c != "Rings"]].copy(),
            "y": pd.Series(abalone_binary_target),
            "schema_hint": {},
        }
    )

    letter_cols = ["target"] + [f"x{i}" for i in range(1, 17)]
    letter_path = _download_with_cache(
        f"{UCI_BASE_URL}/letter-recognition/letter-recognition.data",
        subdir="letter_recognition",
        filename="letter-recognition.data",
        cache_dir=cache_dir,
        allow_download=allow_uci_download,
    )
    letter_df = pd.read_csv(letter_path, header=None, names=letter_cols)
    letter_df = _strip_object_columns(letter_df)

    letter_classes = np.array(sorted(letter_df["target"].dropna().unique().tolist()), dtype=object)
    if letter_classes.shape[0] < 2:
        raise ValueError("Letter Recognition dataset must have at least two classes.")
    class_rng = np.random.default_rng(int(binary_seed))
    chosen_classes = class_rng.choice(letter_classes, size=2, replace=False)
    chosen_classes = np.sort(chosen_classes)

    letter_two_class_df = letter_df[letter_df["target"].isin(chosen_classes)].reset_index(drop=True)
    letter_binary_target = np.where(
        letter_two_class_df["target"].to_numpy() == chosen_classes[0],
        f"letter_{chosen_classes[0]}",
        f"letter_{chosen_classes[1]}",
    )
    datasets.append(
        {
            "name": f"uci_letter_recognition_{chosen_classes[0]}_vs_{chosen_classes[1]}",
            "source": "uci:letter-recognition",
            "X": letter_two_class_df[[f"x{i}" for i in range(1, 17)]].copy(),
            "y": pd.Series(letter_binary_target),
            "schema_hint": {"force_all_continuous": True},
        }
    )

    iranian_churn_zip_path = _download_with_cache(
        "https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip",
        subdir="iranian_churn",
        filename="iranian+churn+dataset.zip",
        cache_dir=cache_dir,
        allow_download=allow_uci_download,
    )
    iranian_churn_df = _read_uci_table_from_zip(iranian_churn_zip_path)
    iranian_churn_df = _strip_object_columns(iranian_churn_df)

    def _norm_name(col_name: object) -> str:
        text = str(col_name).strip().lower()
        text = "".join(ch if ch.isalnum() else "_" for ch in text)
        while "__" in text:
            text = text.replace("__", "_")
        return text.strip("_")

    iranian_norm_to_col = {_norm_name(col): col for col in iranian_churn_df.columns}
    iranian_target_col = next(
        (iranian_norm_to_col[name] for name in ["churn", "target", "y"] if name in iranian_norm_to_col),
        None,
    )
    if iranian_target_col is None:
        raise ValueError("Could not find 'churn' target column in Iranian churn dataset.")

    iranian_target_raw = pd.Series(iranian_churn_df[iranian_target_col])
    iranian_target = pd.to_numeric(iranian_target_raw, errors="coerce")
    if int(iranian_target.nunique(dropna=True)) != 2:
        target_str = iranian_target_raw.astype("string").str.strip().str.lower()
        mapping = {
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "churn": 1,
            "non_churn": 0,
            "non-churn": 0,
        }
        iranian_target = target_str.map(mapping)

    iranian_valid = iranian_target.notna()
    iranian_churn_df = iranian_churn_df.loc[iranian_valid].reset_index(drop=True)
    iranian_target = iranian_target.loc[iranian_valid].astype(np.int64).reset_index(drop=True)

    if int(iranian_target.nunique(dropna=True)) != 2:
        raise ValueError(
            f"Iranian churn target is not binary after cleaning. num_classes={int(iranian_target.nunique(dropna=True))}"
        )

    iranian_feature_cols = [col for col in iranian_churn_df.columns if col != iranian_target_col]
    iranian_force_categorical = [
        iranian_norm_to_col[name]
        for name in ["complains", "status"]
        if name in iranian_norm_to_col and iranian_norm_to_col[name] in iranian_feature_cols
    ]
    iranian_force_continuous = [
        col for col in iranian_feature_cols if col not in set(iranian_force_categorical)
    ]
    datasets.append(
        {
            "name": "uci_iranian_churn_mixed",
            "source": "uci:iranian-churn-dataset",
            "X": iranian_churn_df[iranian_feature_cols].copy(),
            "y": iranian_target.copy(),
            "schema_hint": {
                "force_categorical_cols": iranian_force_categorical,
                "force_continuous_cols": iranian_force_continuous,
            },
        }
    )

    wine_quality_zip_path = _download_with_cache(
        "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
        subdir="wine_quality",
        filename="wine+quality.zip",
        cache_dir=cache_dir,
        allow_download=allow_uci_download,
    )

    with zipfile.ZipFile(wine_quality_zip_path, "r") as wine_zip:
        wine_members = [
            name
            for name in wine_zip.namelist()
            if not name.endswith("/") and name.lower().endswith(".csv") and "winequality" in name.lower()
        ]
        if len(wine_members) == 0:
            raise FileNotFoundError("Could not find winequality CSV files in wine quality zip archive.")

        wine_frames = []
        for member in sorted(wine_members):
            with wine_zip.open(member) as f:
                wine_i = pd.read_csv(f, sep=";")
            wine_i = _strip_object_columns(wine_i)

            if "color" not in wine_i.columns:
                low_member = member.lower()
                if "red" in low_member:
                    wine_i["color"] = "red"
                elif "white" in low_member:
                    wine_i["color"] = "white"
            wine_frames.append(wine_i)

    wine_df = pd.concat(wine_frames, ignore_index=True)
    wine_df = _strip_object_columns(wine_df)

    if "quality" not in wine_df.columns:
        raise ValueError("Wine quality dataset is missing target column 'quality'.")

    quality_series = pd.to_numeric(wine_df["quality"], errors="coerce")
    valid_quality = quality_series.notna()
    wine_df = wine_df.loc[valid_quality].reset_index(drop=True)
    quality_series = quality_series.loc[valid_quality].reset_index(drop=True)

    unique_quality = np.array(sorted(quality_series.unique().tolist()))
    best_cutoff = None
    best_score = (-1, -1)
    for cutoff in unique_quality:
        pos_n = int((quality_series >= cutoff).sum())
        neg_n = int((quality_series < cutoff).sum())
        if pos_n == 0 or neg_n == 0:
            continue
        score = (min(pos_n, neg_n), -abs(pos_n - neg_n))
        if score > best_score:
            best_score = score
            best_cutoff = int(cutoff)

    if best_cutoff is None:
        raise ValueError("Could not determine a balanced cutoff for wine quality.")

    wine_binary_target = np.where(
        quality_series >= best_cutoff,
        f"quality_ge_{best_cutoff}",
        f"quality_lt_{best_cutoff}",
    )

    wine_feature_cols = [col for col in wine_df.columns if col not in {"quality", "color"}]
    datasets.append(
        {
            "name": f"uci_wine_quality_binary_cutoff_{best_cutoff}",
            "source": "uci:wine-quality",
            "X": wine_df[wine_feature_cols].copy(),
            "y": pd.Series(wine_binary_target),
            "schema_hint": {"force_all_continuous": True},
        }
    )

    return datasets


def infer_feature_schema(df: pd.DataFrame, schema_hint: Optional[Dict] = None) -> Dict[str, str]:
    schema_hint = schema_hint or {}
    force_all_categorical = bool(schema_hint.get("force_all_categorical", False))
    force_all_continuous = bool(schema_hint.get("force_all_continuous", False))
    force_categorical_cols = set(schema_hint.get("force_categorical_cols", []))
    force_continuous_cols = set(schema_hint.get("force_continuous_cols", []))

    if force_all_categorical and force_all_continuous:
        raise ValueError("Cannot force all features to both categorical and continuous.")
    overlap = force_categorical_cols & force_continuous_cols
    if len(overlap) > 0:
        raise ValueError(f"Columns listed as both categorical and continuous: {sorted(overlap)}")

    schema: Dict[str, str] = {}
    for col in df.columns:
        s = df[col]
        if force_all_categorical:
            schema[col] = "categorical"
            continue
        if force_all_continuous:
            schema[col] = "continuous"
            continue
        if col in force_continuous_cols:
            schema[col] = "continuous"
            continue
        if col in force_categorical_cols:
            schema[col] = "categorical"
            continue

        if (
            pd.api.types.is_object_dtype(s)
            or isinstance(getattr(s, "dtype", None), pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(s)
        ):
            schema[col] = "categorical"
        elif pd.api.types.is_integer_dtype(s) and int(s.nunique(dropna=True)) <= 20:
            schema[col] = "categorical"
        else:
            schema[col] = "continuous"
    return schema


def encode_dataset_with_schema(
    df: pd.DataFrame,
    schema: Dict[str, str],
    max_categorical_classes: int,
):
    encoded = pd.DataFrame(index=df.index)
    metadata_rows = []

    for col in df.columns:
        kind = schema[col]
        s = df[col]
        raw_unique = int(pd.Series(s).nunique(dropna=True))

        if kind == "categorical":
            s_obj = pd.Series(s, copy=False)
            s_obj = s_obj.where(s_obj.notna(), "__MISSING__").astype("string")

            counts = s_obj.value_counts(dropna=False)
            if counts.shape[0] > max_categorical_classes:
                keep_n = max(1, max_categorical_classes - 1)
                keep_values = set(counts.index[:keep_n].tolist())
                s_obj = s_obj.where(s_obj.isin(keep_values), "__OTHER__")

            cat = pd.Categorical(s_obj.astype(str))
            codes = cat.codes.astype(np.int64)
            cardinality = int(len(cat.categories))

            encoded[col] = codes.astype(np.float32)
            metadata_rows.append(
                {
                    "feature": col,
                    "feature_type": "categorical",
                    "raw_unique_values": raw_unique,
                    "cardinality": cardinality,
                }
            )
        else:
            s_num = pd.to_numeric(s, errors="coerce")
            fill_value = float(s_num.median()) if s_num.notna().any() else 0.0
            s_num = s_num.fillna(fill_value).astype(np.float32)

            encoded[col] = s_num
            metadata_rows.append(
                {
                    "feature": col,
                    "feature_type": "continuous",
                    "raw_unique_values": raw_unique,
                    "cardinality": 1,
                }
            )

    feature_metadata = pd.DataFrame(metadata_rows)
    feature_is_categorical = (feature_metadata["feature_type"].to_numpy() == "categorical")
    feature_cardinalities = feature_metadata["cardinality"].to_numpy(dtype=np.int64)

    X_np = encoded.to_numpy(dtype=np.float32)
    return X_np, feature_is_categorical, feature_cardinalities, feature_metadata


def drop_high_cardinality_categorical_features(
    df: pd.DataFrame,
    schema: Dict[str, str],
    max_allowed_cardinality: int = 10,
):
    drop_cols = []
    for col in df.columns:
        if schema.get(col) != "categorical":
            continue
        n_unique = int(pd.Series(df[col]).nunique(dropna=True))
        if n_unique > max_allowed_cardinality:
            drop_cols.append(col)

    if len(drop_cols) > 0:
        df = df.drop(columns=drop_cols)

    schema = {col: schema[col] for col in df.columns if col in schema}
    return df, schema, drop_cols


def prepare_dataset(record: Dict, max_categorical_classes: int):
    X_raw = record["X"].reset_index(drop=True)
    y_raw = pd.Series(record["y"]).reset_index(drop=True)

    valid = y_raw.notna()
    X_raw = X_raw.loc[valid].reset_index(drop=True)
    y_raw = y_raw.loc[valid].reset_index(drop=True)

    if y_raw.nunique(dropna=True) != 2:
        raise ValueError(f"Dataset '{record['name']}' is not binary after cleaning.")

    schema = infer_feature_schema(X_raw, schema_hint=record.get("schema_hint"))
    X_filtered, schema, dropped_cols = drop_high_cardinality_categorical_features(
        X_raw,
        schema,
        max_allowed_cardinality=10,
    )
    if X_filtered.shape[1] == 0:
        raise ValueError(
            f"Dataset '{record['name']}' has no features after removing categorical columns with >10 classes."
        )

    X_np, feature_is_cat, feature_card, feature_meta = encode_dataset_with_schema(
        X_filtered,
        schema,
        max_categorical_classes=max_categorical_classes,
    )
    if len(dropped_cols) > 0:
        print(f"[info] {record['name']}: dropped high-cardinality categorical columns: {sorted(dropped_cols)}")

    return {
        "name": record["name"],
        "source": record["source"],
        "X": X_np,
        "y": y_raw.to_numpy(),
        "feature_is_categorical": feature_is_cat,
        "feature_cardinalities": feature_card,
        "feature_metadata": feature_meta,
    }


def build_pu_task(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    max_positive_size: int,
    unlabeled_labeled_positive_ratio: tuple[int, int],
    outlier_rate: float,
):
    labels = np.unique(y)
    if labels.shape[0] != 2:
        return None

    positive_label = rng.choice(labels)
    pos_idx = np.where(y == positive_label)[0]
    neg_idx = np.where(y != positive_label)[0]

    if len(pos_idx) < 2 or len(neg_idx) < 1:
        return None

    selected_pos_n = int(min(max_positive_size, len(pos_idx)))
    selected_pos_idx = rng.choice(pos_idx, size=selected_pos_n, replace=False)

    u_ratio, l_ratio = unlabeled_labeled_positive_ratio
    if u_ratio < 0 or l_ratio <= 0 or (u_ratio + l_ratio) <= 0:
        raise ValueError("UNLABELED_LABELED_POSITIVE_RATIO must be (u, l) with u>=0, l>0.")

    unlabeled_pos_n = int(round(selected_pos_n * (float(u_ratio) / float(u_ratio + l_ratio))))
    unlabeled_pos_n = int(np.clip(unlabeled_pos_n, 1, selected_pos_n - 1))
    labeled_pos_n = selected_pos_n - unlabeled_pos_n
    if labeled_pos_n <= 0 or unlabeled_pos_n <= 0:
        return None

    labeled_pos_idx = rng.choice(selected_pos_idx, size=labeled_pos_n, replace=False)
    unlabeled_pos_idx = np.setdiff1d(selected_pos_idx, labeled_pos_idx, assume_unique=False)

    neg_needed = int(round(unlabeled_pos_n * outlier_rate / max(1e-8, 1.0 - outlier_rate)))
    if outlier_rate > 0.0 and unlabeled_pos_n > 0:
        neg_needed = max(1, neg_needed)
    neg_needed = min(neg_needed, len(neg_idx))
    if neg_needed <= 0:
        return None

    unlabeled_neg_idx = rng.choice(neg_idx, size=neg_needed, replace=False)

    unlabeled_idx = np.concatenate([unlabeled_pos_idx, unlabeled_neg_idx])
    unlabeled_y = np.concatenate(
        [
            np.zeros(unlabeled_pos_idx.shape[0], dtype=np.int64),
            np.ones(unlabeled_neg_idx.shape[0], dtype=np.int64),
        ]
    )
    perm = rng.permutation(unlabeled_idx.shape[0])
    unlabeled_idx = unlabeled_idx[perm]
    unlabeled_y = unlabeled_y[perm]

    labeled_perm = rng.permutation(labeled_pos_idx.shape[0])
    labeled_pos_idx = labeled_pos_idx[labeled_perm]

    X_task = np.concatenate([X[labeled_pos_idx], X[unlabeled_idx]], axis=0).astype(np.float32)
    y_train = np.zeros(labeled_pos_idx.shape[0], dtype=np.float32)

    return {
        "X": X_task,
        "y_train": y_train,
        "y_test": unlabeled_y,
        "train_size": int(labeled_pos_idx.shape[0]),
        "num_unlabeled_inliers": int(unlabeled_pos_idx.shape[0]),
        "num_unlabeled_outliers": int(unlabeled_neg_idx.shape[0]),
        "positive_label": str(positive_label),
    }


def fpr_at_fixed_tpr(y_true: np.ndarray, outlier_score: np.ndarray, target_tpr: float = 0.95) -> float:
    if np.unique(y_true).shape[0] < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, outlier_score)
    valid = np.where(tpr >= target_tpr)[0]
    if valid.size == 0:
        return 1.0
    return float(np.min(fpr[valid]))


@torch.no_grad()
def evaluate_single_pu_task(
    model,
    task: Dict,
    feature_is_categorical: np.ndarray,
    feature_cardinalities: np.ndarray,
    device: str,
    supports_categorical: bool,
) -> Dict[str, float]:
    x = torch.from_numpy(task["X"]).unsqueeze(0).to(device=device, dtype=torch.float32)
    y_train = torch.from_numpy(task["y_train"]).unsqueeze(0).to(device=device, dtype=torch.float32)

    if supports_categorical:
        feature_is_cat_t = torch.from_numpy(feature_is_categorical).unsqueeze(0).to(device=device, dtype=torch.bool)
        feature_card_t = torch.from_numpy(feature_cardinalities).unsqueeze(0).to(device=device, dtype=torch.long)
        logits = model(
            (x, y_train),
            train_test_split_index=task["train_size"],
            feature_is_categorical=feature_is_cat_t,
            feature_cardinalities=feature_card_t,
        ).squeeze(0)
    else:
        logits = model(
            (x, y_train),
            train_test_split_index=task["train_size"],
        ).squeeze(0)

    logits_np = logits.detach().cpu().numpy()
    y_true = task["y_test"].astype(np.int64)
    y_pred = np.argmax(logits_np, axis=1)

    outlier_score = logits_np[:, 1]

    binary_ready = np.unique(y_true).shape[0] == 2
    outlier_mean = float(np.mean(outlier_score[y_true == 1])) if np.any(y_true == 1) else float("nan")
    inlier_mean = float(np.mean(outlier_score[y_true == 0])) if np.any(y_true == 0) else float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, outlier_score)) if binary_ready else float("nan"),
        "average_precision": float(average_precision_score(y_true, outlier_score)) if binary_ready else float("nan"),
        "fpr_at_tpr_0_80": float(fpr_at_fixed_tpr(y_true, outlier_score, target_tpr=0.80)),
        "fpr_at_tpr_0_90": float(fpr_at_fixed_tpr(y_true, outlier_score, target_tpr=0.90)),
        "fpr_at_tpr_0_95": float(fpr_at_fixed_tpr(y_true, outlier_score, target_tpr=0.95)),
        "outlier_score_gap": float(outlier_mean - inlier_mean)
        if (not np.isnan(outlier_mean) and not np.isnan(inlier_mean))
        else float("nan"),
    }


def summarize_profile(prepared_datasets: list[Dict]) -> pd.DataFrame:
    profile_rows = []
    for dataset in prepared_datasets:
        feature_meta = dataset["feature_metadata"]
        num_cat = int((feature_meta["feature_type"] == "categorical").sum())
        num_cont = int((feature_meta["feature_type"] == "continuous").sum())
        max_card = (
            int(feature_meta.loc[feature_meta["feature_type"] == "categorical", "cardinality"].max())
            if num_cat > 0
            else 1
        )
        profile_rows.append(
            {
                "dataset": dataset["name"],
                "source": dataset["source"],
                "rows": int(dataset["X"].shape[0]),
                "features": int(dataset["X"].shape[1]),
                "continuous_features": num_cont,
                "categorical_features": num_cat,
                "max_categorical_cardinality": max_card,
            }
        )
    return pd.DataFrame(profile_rows)


def aggregate_results(replicate_results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if replicate_results_df.empty:
        empty_summary = pd.DataFrame(columns=["dataset", "source", "replicates"] + METRIC_COLUMNS)
        empty_composition = pd.DataFrame(
            columns=[
                "dataset",
                "source",
                "replicates",
                "true_positive_only_sample_size",
                "true_unlabeled_to_labeled_positive_ratio",
                "true_outlier_rate",
            ]
        )
        return empty_summary, empty_summary.copy(), empty_composition

    metrics_by_model_df = (
        replicate_results_df.groupby(["dataset", "source", "model_name"], as_index=False)
        .agg(
            replicates=("replicate", "count"),
            accuracy=("accuracy", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            roc_auc=("roc_auc", "mean"),
            average_precision=("average_precision", "mean"),
            fpr_at_tpr_0_80=("fpr_at_tpr_0_80", "mean"),
            fpr_at_tpr_0_90=("fpr_at_tpr_0_90", "mean"),
            fpr_at_tpr_0_95=("fpr_at_tpr_0_95", "mean"),
            outlier_score_gap=("outlier_score_gap", "mean"),
        )
        .sort_values(["dataset", "model_name"])
        .reset_index(drop=True)
    )

    metrics_latest_df = (
        metrics_by_model_df[metrics_by_model_df["model_name"] == "latest"]
        .drop(columns=["model_name"])
        .reset_index(drop=True)
    )

    composition_summary_df = (
        replicate_results_df.groupby(["dataset", "source"], as_index=False)
        .agg(
            replicates=("replicate", "count"),
            true_positive_only_sample_size=("real_positive_only_sample_size", "mean"),
            true_unlabeled_to_labeled_positive_ratio=("real_unlabeled_positive_to_labeled_positive_ratio", "mean"),
            true_outlier_rate=("real_outlier_proportion", "mean"),
        )
        .sort_values("dataset")
        .reset_index(drop=True)
    )
    return metrics_by_model_df, metrics_latest_df, composition_summary_df


def main() -> None:
    args = parse_args()
    repo_root, pretrain_root, package_name = resolve_runtime_paths()
    device = resolve_device(args.device)

    checkpoint_path = args.checkpoint.expanduser() if args.checkpoint is not None else pretrain_root / "pretrained_model" / "latest.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not (0.0 <= args.outlier_rate < 1.0):
        raise ValueError("outlier_rate must satisfy 0 <= outlier_rate < 1.")

    output_dir = args.output_dir.expanduser() if args.output_dir is not None else pretrain_root / "evaluation_outputs"
    cache_dir = resolve_cache_dir(args.cache_dir, pretrain_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_module = importlib.import_module(f"{package_name}.model")
    model_cls = getattr(model_module, "NanoTabPFNPUModel")
    model, _, load_result = load_model(checkpoint_path=checkpoint_path, model_cls=model_cls, device=device)

    benchmark_datasets = get_benchmark_datasets(
        cache_dir=cache_dir,
        allow_uci_download=args.allow_uci_download,
        binary_seed=args.global_seed,
    )
    prepared_datasets = [
        prepare_dataset(dataset, max_categorical_classes=args.max_categorical_classes)
        for dataset in benchmark_datasets
    ]
    profile_df = summarize_profile(prepared_datasets)

    model_specs = [
        {
            "model_name": "latest",
            "model": model,
            "supports_categorical": False,
            "checkpoint_path": str(checkpoint_path),
        }
    ]

    rng_master = np.random.default_rng(args.global_seed)
    replicate_frames = []
    for dataset in prepared_datasets:
        ds_seed = int(rng_master.integers(0, 2**31 - 1))
        ds_rng = np.random.default_rng(ds_seed)
        rows = []
        attempts = 0
        collected_replicates = 0

        while collected_replicates < args.n_replicates and attempts < args.max_attempts_per_dataset:
            attempts += 1
            task = build_pu_task(
                X=dataset["X"],
                y=dataset["y"],
                rng=ds_rng,
                max_positive_size=args.max_positive_size,
                unlabeled_labeled_positive_ratio=(args.unlabeled_positive_ratio, args.labeled_positive_ratio),
                outlier_rate=args.outlier_rate,
            )
            if task is None:
                continue

            unlabeled_total = int(task["num_unlabeled_inliers"] + task["num_unlabeled_outliers"])
            real_outlier_proportion = (
                float(task["num_unlabeled_outliers"]) / float(unlabeled_total) if unlabeled_total > 0 else float("nan")
            )
            real_unlabeled_positive_to_labeled_positive_ratio = (
                float(task["num_unlabeled_inliers"]) / float(task["train_size"])
                if int(task["train_size"]) > 0
                else float("nan")
            )
            real_positive_only_sample_size = int(task["train_size"] + task["num_unlabeled_inliers"])

            for model_spec in model_specs:
                metric = evaluate_single_pu_task(
                    model=model_spec["model"],
                    task=task,
                    feature_is_categorical=dataset["feature_is_categorical"],
                    feature_cardinalities=dataset["feature_cardinalities"],
                    device=device,
                    supports_categorical=bool(model_spec["supports_categorical"]),
                )
                metric.update(
                    {
                        "model_name": model_spec["model_name"],
                        "dataset": dataset["name"],
                        "source": dataset["source"],
                        "replicate": collected_replicates + 1,
                        "attempt": attempts,
                        "positive_label": task["positive_label"],
                        "labeled_positive_size": int(task["train_size"]),
                        "real_labeled_positive_size": int(task["train_size"]),
                        "real_positive_only_sample_size": real_positive_only_sample_size,
                        "unlabeled_inlier_size": int(task["num_unlabeled_inliers"]),
                        "unlabeled_outlier_size": int(task["num_unlabeled_outliers"]),
                        "real_outlier_proportion": real_outlier_proportion,
                        "real_unlabeled_positive_to_labeled_positive_ratio": real_unlabeled_positive_to_labeled_positive_ratio,
                    }
                )
                rows.append(metric)

            collected_replicates += 1

        if collected_replicates < args.n_replicates:
            print(
                f"[warn] dataset={dataset['name']} collected {collected_replicates} replicates "
                f"within {args.max_attempts_per_dataset} attempts."
            )

        replicate_frames.append(pd.DataFrame(rows))

    replicate_results_df = (
        pd.concat(replicate_frames, ignore_index=True) if len(replicate_frames) > 0 else pd.DataFrame()
    )
    metrics_by_model_df, metrics_latest_df, composition_summary_df = aggregate_results(replicate_results_df)

    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"eval_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary_metrics.csv"
    summary_latest_path = run_dir / "summary_metrics_latest.csv"
    metrics_by_model_path = run_dir / "metrics_by_model.csv"
    replicate_path = run_dir / "replicate_metrics.csv"
    profile_path = run_dir / "dataset_feature_profile.csv"
    composition_path = run_dir / "pu_composition_summary.csv"
    config_path = run_dir / "run_config.json"

    metrics_latest_df.to_csv(summary_path, index=False)
    metrics_latest_df.to_csv(summary_latest_path, index=False)
    metrics_by_model_df.to_csv(metrics_by_model_path, index=False)
    replicate_results_df.to_csv(replicate_path, index=False)
    profile_df.to_csv(profile_path, index=False)
    composition_summary_df.to_csv(composition_path, index=False)

    for dataset in prepared_datasets:
        safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in dataset["name"])
        dataset["feature_metadata"].to_csv(run_dir / f"feature_metadata_{safe_name}.csv", index=False)

    run_config = {
        "checkpoint_path": _public_path_label(checkpoint_path, repo_root),
        "device": device,
        "allow_uci_download": args.allow_uci_download,
        "n_replicates": args.n_replicates,
        "max_attempts_per_dataset": args.max_attempts_per_dataset,
        "max_positive_size": args.max_positive_size,
        "unlabeled_labeled_positive_ratio": [args.unlabeled_positive_ratio, args.labeled_positive_ratio],
        "outlier_rate": args.outlier_rate,
        "max_categorical_classes": args.max_categorical_classes,
        "global_seed": args.global_seed,
        "cache_dir": _public_path_label(cache_dir, repo_root),
        "load_missing_keys": list(load_result.missing_keys),
        "load_unexpected_keys": list(load_result.unexpected_keys),
    }
    config_path.write_text(json.dumps(run_config, indent=2))

    print(f"repo_root={repo_root}")
    print(f"pretrain_root={pretrain_root}")
    print(f"device={device}")
    print(f"cache_dir={cache_dir}")
    print(f"Loaded latest model: missing_keys={len(load_result.missing_keys)}, unexpected_keys={len(load_result.unexpected_keys)}")
    print(f"Saved evaluation outputs to: {run_dir}")
    print(f"- {summary_path.name}")
    print(f"- {summary_latest_path.name}")
    print(f"- {metrics_by_model_path.name}")
    print(f"- {replicate_path.name}")
    print(f"- {profile_path.name}")
    print(f"- {composition_path.name}")


if __name__ == "__main__":
    main()
