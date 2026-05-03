# In-Context Learning for Positive-Unlabeled Classification and Outlier Detection

[![test](https://github.com/qinglong-tian/puicl/actions/workflows/tests.yml/badge.svg)](https://github.com/qinglong-tian/puicl/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
[![license](https://img.shields.io/github/license/qinglong-tian/puicl)](LICENSE)

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
pip install git+https://github.com/qinglong-tian/puicl.git
```

If you also want to run the benchmark evaluator in this repository, install the optional evaluation dependencies:

```bash
pip install .[eval]
```

## Quick Start

The example below uses the UCI Banknote Authentication dataset, converts it into a positive-unlabeled task, and scores the unlabeled pool with the packaged pretrained model.

```bash
python - <<'PY'
import numpy as np

from puicl import load_pretrained_model
from puicl.utils import load_banknote_dataset, make_pu_task


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

Example output from a run of the snippet above on March 11, 2026:

```text
labeled positives: 64
unlabeled rows: 400
accuracy: 0.9450
confusion matrix:
[[178  22]
 [  0 200]]
first 10 outlier probabilities:
[0.9975 0.0136 0.9183 0.0116 0.0107 0.9333 0.019  0.9999 0.0126 0.0103]
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

Latest run in this repository (March 11, 2026; bundled `latest.pt`; `device=mps`; `global_seed=42`; `n_replicates=10`) used the example command above and wrote outputs to `evaluation_outputs/eval_20260311_132907/`.

Macro-average metrics across the 13 benchmark datasets from that run:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8553 |
| Balanced accuracy | 0.7866 |
| ROC AUC | 0.8744 |
| Average precision | 0.7332 |
| FPR@TPR=0.80 | 0.2118 |
| FPR@TPR=0.90 | 0.3287 |
| FPR@TPR=0.95 | 0.4336 |
| Outlier score gap | 1.9165 |

Per-dataset summary from `summary_metrics_latest.csv`:

| Dataset | Accuracy | Balanced accuracy | ROC AUC | Average precision |
| --- | ---: | ---: | ---: | ---: |
| Abalone | 0.8508 | 0.7732 | 0.8648 | 0.6952 |
| Adult | 0.8342 | 0.6549 | 0.7918 | 0.5836 |
| Banknote authentication | 0.9844 | 0.9902 | 0.9992 | 0.9968 |
| Car evaluation | 0.6226 | 0.6983 | 0.8859 | 0.6983 |
| Default credit card clients | 0.7646 | 0.5739 | 0.6235 | 0.3255 |
| Iranian churn | 0.8930 | 0.7986 | 0.9107 | 0.8073 |
| Letter recognition (C vs U) | 0.9724 | 0.9636 | 0.9946 | 0.9816 |
| MAGIC gamma | 0.8288 | 0.6759 | 0.8052 | 0.5981 |
| Mushroom | 0.9476 | 0.9309 | 0.9765 | 0.9523 |
| Rice Cammeo/Osmancik | 0.9346 | 0.8778 | 0.9691 | 0.9113 |
| Spambase | 0.7668 | 0.7909 | 0.8756 | 0.6754 |
| WDBC | 0.9291 | 0.8978 | 0.9598 | 0.8862 |
| Wine quality (cutoff 6) | 0.7894 | 0.6003 | 0.7108 | 0.4205 |

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
