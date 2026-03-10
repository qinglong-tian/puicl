from __future__ import annotations

import argparse
import ast
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


DEFAULT_REFERENCE_NOTEBOOK = Path(__file__).resolve().parent / "evaluate_pretrained_model_reference.ipynb"

REFERENCE_NAMES = {
    "UCI_BASE_URL",
    "_download_with_cache",
    "_strip_object_columns",
    "_read_uci_table_from_zip",
    "get_benchmark_datasets",
    "infer_feature_schema",
    "encode_dataset_with_schema",
    "drop_high_cardinality_categorical_features",
    "prepare_dataset",
    "build_pu_task",
    "fpr_at_fixed_tpr",
    "evaluate_single_pu_task",
}

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
            "Evaluate pretrained_model/latest.pt using the exact dataset definitions, "
            "PU conversion, and metrics from the reference evaluation notebook."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--reference-notebook", type=Path, default=DEFAULT_REFERENCE_NOTEBOOK)
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


def resolve_cache_dir(requested: Optional[Path], reference_notebook: Path, pretrain_root: Path) -> Path:
    if requested is not None:
        return requested.expanduser()

    reference_cache = reference_notebook.expanduser().resolve().parent / ".cache"
    if reference_cache.exists():
        return reference_cache
    return pretrain_root / ".cache"


def load_reference_namespace(reference_notebook: Path, cache_dir: Path) -> Dict[str, object]:
    notebook_path = reference_notebook.expanduser().resolve()
    if not notebook_path.exists():
        raise FileNotFoundError(f"Reference notebook not found: {notebook_path}")

    notebook = json.loads(notebook_path.read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    tree = ast.parse(source, filename=str(notebook_path))

    selected_nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & REFERENCE_NAMES:
                selected_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in REFERENCE_NAMES:
            selected_nodes.append(node)

    extracted = ast.Module(body=selected_nodes, type_ignores=[])
    namespace: Dict[str, object] = {
        "__builtins__": __builtins__,
        "np": np,
        "pd": pd,
        "torch": torch,
        "arff": arff,
        "BytesIO": BytesIO,
        "TextIOWrapper": TextIOWrapper,
        "Path": Path,
        "Dict": Dict,
        "Optional": Optional,
        "zipfile": zipfile,
        "urlretrieve": urlretrieve,
        "accuracy_score": accuracy_score,
        "average_precision_score": average_precision_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "roc_auc_score": roc_auc_score,
        "roc_curve": roc_curve,
        "CACHE_DIR": cache_dir,
    }
    exec(compile(extracted, str(notebook_path), "exec"), namespace)

    missing = sorted(name for name in REFERENCE_NAMES if name not in namespace)
    if missing:
        raise RuntimeError(f"Failed to extract required reference definitions: {missing}")
    return namespace


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

    output_dir = (args.output_dir.expanduser() if args.output_dir is not None else pretrain_root / "evaluation_outputs")
    cache_dir = resolve_cache_dir(args.cache_dir, args.reference_notebook, pretrain_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_module = importlib.import_module(f"{package_name}.model")
    model_cls = getattr(model_module, "NanoTabPFNPUModel")
    model, _, load_result = load_model(checkpoint_path=checkpoint_path, model_cls=model_cls, device=device)

    ref_ns = load_reference_namespace(args.reference_notebook, cache_dir=cache_dir)
    get_benchmark_datasets = ref_ns["get_benchmark_datasets"]
    prepare_dataset = ref_ns["prepare_dataset"]
    build_pu_task = ref_ns["build_pu_task"]
    evaluate_single_pu_task = ref_ns["evaluate_single_pu_task"]

    benchmark_datasets = get_benchmark_datasets(
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
        "checkpoint_path": str(checkpoint_path),
        "reference_notebook": str(args.reference_notebook.expanduser()),
        "device": device,
        "allow_uci_download": args.allow_uci_download,
        "n_replicates": args.n_replicates,
        "max_attempts_per_dataset": args.max_attempts_per_dataset,
        "max_positive_size": args.max_positive_size,
        "unlabeled_labeled_positive_ratio": [args.unlabeled_positive_ratio, args.labeled_positive_ratio],
        "outlier_rate": args.outlier_rate,
        "max_categorical_classes": args.max_categorical_classes,
        "global_seed": args.global_seed,
        "cache_dir": str(cache_dir),
        "load_missing_keys": list(load_result.missing_keys),
        "load_unexpected_keys": list(load_result.unexpected_keys),
    }
    config_path.write_text(json.dumps(run_config, indent=2))

    print(f"repo_root={repo_root}")
    print(f"pretrain_root={pretrain_root}")
    print(f"device={device}")
    print(f"reference_notebook={args.reference_notebook.expanduser()}")
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
