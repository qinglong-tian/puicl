# In-Context Learning for Positive-Unlabeled Classification and Outlier Detection

[![test](https://github.com/qinglong-tian/PU_ICL_Code/actions/workflows/tests.yml/badge.svg)](https://github.com/qinglong-tian/PU_ICL_Code/actions/workflows/tests.yml)
[![pypi package](https://img.shields.io/pypi/v/puicl)](https://pypi.org/project/puicl/)
[![downloads](https://static.pepy.tech/badge/puicl/month)](https://pepy.tech/project/puicl)

This repository contains two things:

- an installable Python package, `puicl`, for inference with the bundled pretrained model
- the research code used for pretraining and benchmark evaluation

The intended end-user path is:

1. install `puicl`
2. load the packaged checkpoint with `load_pretrained_model()`
3. score unlabeled examples given a set of labeled positives

## Installation

Install from a local checkout:

```bash
pip install .
```

Install directly from GitHub:

```bash
pip install git+https://github.com/qinglong-tian/PU_ICL_Code.git
```

If you also want to run the benchmark evaluator in this repository, install the optional evaluation dependencies:

```bash
pip install .[eval]
```

## Quick Start

The example below uses the UCI Banknote Authentication dataset, converts it into a positive-unlabeled task, and scores the unlabeled pool with the packaged pretrained model.

```bash
python - <<'PY'
from io import BytesIO
from urllib.request import urlopen
import zipfile

import numpy as np

from puicl import load_pretrained_model


def load_banknote_dataset() -> tuple[np.ndarray, np.ndarray]:
    url = "https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip"
    with urlopen(url) as response:
        raw = response.read()
    with zipfile.ZipFile(BytesIO(raw)) as zf:
        with zf.open("data_banknote_authentication.txt") as f:
            data = np.loadtxt(f, delimiter=",", dtype=np.float32)
    x = data[:, :4]
    y = data[:, 4].astype(np.int64)
    return x, y


def make_pu_task(
    x: np.ndarray,
    y: np.ndarray,
    *,
    positive_label: int = 0,
    labeled_positive_size: int = 64,
    unlabeled_positive_size: int = 200,
    unlabeled_outlier_size: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    pos_idx = np.where(y == positive_label)[0]
    neg_idx = np.where(y != positive_label)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    labeled_idx = pos_idx[:labeled_positive_size]
    unlabeled_pos_idx = pos_idx[
        labeled_positive_size : labeled_positive_size + unlabeled_positive_size
    ]
    unlabeled_neg_idx = neg_idx[:unlabeled_outlier_size]

    x_labeled = x[labeled_idx]
    x_unlabeled = np.concatenate([x[unlabeled_pos_idx], x[unlabeled_neg_idx]], axis=0)
    y_unlabeled_true = np.concatenate(
        [
            np.zeros(len(unlabeled_pos_idx), dtype=np.int64),
            np.ones(len(unlabeled_neg_idx), dtype=np.int64),
        ],
        axis=0,
    )

    perm = rng.permutation(len(y_unlabeled_true))
    x_unlabeled = x_unlabeled[perm]
    y_unlabeled_true = y_unlabeled_true[perm]
    return x_labeled, x_unlabeled, y_unlabeled_true


x, y = load_banknote_dataset()
x_labeled, x_unlabeled, y_unlabeled_true = make_pu_task(x, y)

model = load_pretrained_model(device="cpu")
outlier_prob = model.score_unlabeled(x_labeled, x_unlabeled).numpy()
y_pred = (outlier_prob >= 0.5).astype(np.int64)

tn = int(np.sum((y_unlabeled_true == 0) & (y_pred == 0)))
fp = int(np.sum((y_unlabeled_true == 0) & (y_pred == 1)))
fn = int(np.sum((y_unlabeled_true == 1) & (y_pred == 0)))
tp = int(np.sum((y_unlabeled_true == 1) & (y_pred == 1)))
accuracy = float(np.mean(y_pred == y_unlabeled_true))

print(f"labeled positives: {x_labeled.shape[0]}")
print(f"unlabeled rows: {x_unlabeled.shape[0]}")
print(f"accuracy: {accuracy:.4f}")
print("confusion matrix:")
print(np.array([[tn, fp], [fn, tp]]))
print("first 10 outlier probabilities:")
print(np.round(outlier_prob[:10], 4))
PY
```

The key inference convention is:

- the model sees a prefix of labeled positive rows
- every row after that prefix is treated as unlabeled and receives an outlier probability
- output class `1` is the outlier score used by the benchmark code

## Public API

The installable package is under [src/puicl](src/puicl).

Main entry points:

- `from puicl import load_pretrained_model`
- `from puicl import PUICLModel`
- `from puicl import NanoTabPFNPUModel`

The `PUICLModel` wrapper provides:

- `score_unlabeled(labeled_positive_features, unlabeled_features)`
- `predict_logits(x_task, train_test_split_index=..., y_train=...)`
- `predict_proba(x_task, train_test_split_index=..., y_train=...)`
- `predict_labels(x_task, train_test_split_index=..., y_train=...)`

You can also override the bundled checkpoint:

```python
from puicl import load_pretrained_model

model = load_pretrained_model(checkpoint="/path/to/latest.pt", device="cuda")
```

## Package Contents

The packaged checkpoint is bundled as package data:

- [src/puicl/checkpoints/latest.pt](src/puicl/checkpoints/latest.pt)

The core model implementation used by the package is:

- [src/puicl/model.py](src/puicl/model.py)

The high-level inference wrapper is:

- [src/puicl/inference.py](src/puicl/inference.py)

## Research Code

The repository also includes the research pipeline used to produce and evaluate the checkpoint:

- [train/](train): pretraining configuration, trainer, and launchers
- [simplified_prior/](simplified_prior): synthetic prior and curriculum logic
- [data/](data): padded-batch generation
- [evaluate_pretrained_model.py](evaluate_pretrained_model.py): standalone benchmark evaluator
- [run_pretrain_two_phase_hpc_v2.sbatch](run_pretrain_two_phase_hpc_v2.sbatch): Slurm launcher template

Current default training configuration:

- model: 6-layer transformer, embedding size `128`, 8 attention heads, MLP hidden size `256`
- curriculum: `100` stages with `1000` steps per stage
- tail phase: additional `25000` steps on the final-stage distribution
- optimization defaults: batch size `24`, base LR `1.6e-4`, minimum LR `1.6e-5`
- synthetic task ranges: `5..24` features and `100..300` labeled positives

The public Slurm script omits site-specific account and email directives on purpose. Add those locally if your cluster requires them.

## Evaluation

The benchmark evaluator is a repository script, not a packaged CLI. It uses:

- a fixed set of binary tabular benchmark datasets
- a fixed PU conversion protocol
- fixed metrics including accuracy, balanced accuracy, ROC AUC, average precision, and FPR-at-TPR targets

Example:

```bash
python evaluate_pretrained_model.py \
  --max-positive-size 600 \
  --unlabeled-positive-ratio 2 \
  --labeled-positive-ratio 1 \
  --outlier-rate 0.2
```

Important evaluator arguments:

- `--checkpoint`: checkpoint to evaluate; defaults to the bundled repository checkpoint at `src/puicl/checkpoints/latest.pt`
- `--device`: `auto`, `cpu`, `cuda`, or `mps`
- `--cache-dir`: location for cached UCI files
- `--n-replicates`: number of PU tasks sampled per dataset
- `--max-positive-size`: maximum number of positives used to construct one PU task
- `--unlabeled-positive-ratio` and `--labeled-positive-ratio`: split between unlabeled positives and labeled positives
- `--outlier-rate`: target fraction of negatives inside the unlabeled pool

Evaluation outputs are written to `evaluation_outputs/eval_<timestamp>/`.

## Notes

- The installable `puicl` package is focused on inference with the bundled checkpoint.
- Training and full benchmark evaluation remain repository workflows rather than stable package APIs.
