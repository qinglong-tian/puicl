# In-Context Learning for Positive-Unlabeled Classification and Outlier Detection

Code and experiment artifacts for PU in-context learning on tabular data.

This repository contains:
- an installable `puicl` Python package for inference with the bundled checkpoint
- the PU-adapted TabPFN-style model used in the paper
- synthetic-prior pretraining code and curriculum setup
- an evaluation pipeline on a fixed set of binary tabular benchmarks
- a pretrained checkpoint and example evaluation outputs

## Repository Layout

- `src/puicl/`: installable inference package and bundled checkpoint
- `model.py`: main `NanoTabPFNPUModel`
- `train/`: pretraining config, trainer, schedules, and launch entry points
- `simplified_prior/`: synthetic data generator and curriculum sampler
- `data/`: padded-batch generation for PU pretraining
- `pretrained_model/latest.pt`: repository copy of the pretrained checkpoint
- `evaluate_pretrained_model.py`: benchmark evaluator for the pretrained checkpoint
- `evaluation_outputs/`: saved benchmark runs
- `run_pretrain_two_phase_hpc_v2.sbatch`: two-phase HPC training launcher

## Installation

Install from a local checkout:

```bash
pip install .
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/qinglong-tian/PU_ICL_Code.git
```

Install the evaluation extras if you also want to run the full benchmark script:

```bash
pip install .[eval]
```

## Quick Start

Load the packaged checkpoint and run the full synthetic PU example:

```bash
python - <<'PY'
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from puicl import load_pretrained_model

torch.manual_seed(0)
device = "cpu"  # use "cuda" here if you want GPU inference
model = load_pretrained_model(device=device)

# Synthetic PU task:
# - labeled positives are sampled from one Gaussian
# - unlabeled rows are an explicit mixture of positive and negative Gaussians
train_size = 64
num_unlabeled_pos = 128
num_unlabeled_neg = 128
num_features = 12

positive_mean = torch.full((num_features,), 1.0, device=device)
negative_mean = torch.full((num_features,), -1.0, device=device)

labeled_pos = torch.randn(train_size, num_features, device=device) + positive_mean
unlabeled_pos = torch.randn(num_unlabeled_pos, num_features, device=device) + positive_mean
unlabeled_neg = torch.randn(num_unlabeled_neg, num_features, device=device) + negative_mean

X_unlabeled = torch.cat([unlabeled_pos, unlabeled_neg], dim=0)
y_true = torch.cat(
    [
        torch.zeros(num_unlabeled_pos, dtype=torch.long, device=device),
        torch.ones(num_unlabeled_neg, dtype=torch.long, device=device),
    ],
    dim=0,
)

perm = torch.randperm(X_unlabeled.shape[0], device=device)
X_unlabeled = X_unlabeled[perm]
y_true = y_true[perm]

outlier_prob = model.score_unlabeled(labeled_pos, X_unlabeled)
y_pred = (outlier_prob >= 0.5).to(torch.long)

accuracy = accuracy_score(y_true.numpy(), y_pred.numpy())
auc = roc_auc_score(y_true.numpy(), outlier_prob.numpy())
cm = confusion_matrix(y_true.numpy(), y_pred.numpy())

print(f"num_unlabeled={len(y_true)}")
print(f"accuracy={accuracy:.4f}")
print(f"auc={auc:.4f}")
print(cm)
PY
```

The evaluator remains available as a repository script and is documented below in the evaluation section.

## Model

The main predictive model is a PU-adapted TabPFN-style transformer:
- scalar feature encoder with train-prefix normalization
- PU target encoder with a learned unlabeled token
- transformer blocks with feature-wise and row-wise attention
- 2-class decoder for inlier vs outlier prediction on unlabeled rows

Current default architecture:
- embedding size: `128`
- attention heads: `8`
- transformer layers: `6`
- MLP hidden size: `256`

## Training

The current default pretraining schedule is two-phase:

1. Curriculum phase:
- `100` stages
- `1000` steps per stage

2. Final-stage tail:
- additional `25000` steps
- fixed final-stage data distribution
- reduced learning rate

Current launcher defaults:
- batch size: `24`
- base learning rate: `1.6e-4`
- minimum learning rate: `1.6e-5`
- feature-count range: `5..24`
- positive train-size range: `100..300`

The provided Slurm launcher currently requests:
- `1` node
- `2 x H100` GPUs
- `4` CPU cores
- `16G` RAM
- `02:30:00` wall time

Cluster-specific account and email directives are intentionally omitted from the public script. Add your own site-specific `#SBATCH --account=...` or mail settings locally if your cluster requires them.

Example launch:

```bash
sbatch /path/to/PU_ICL_Code/run_pretrain_two_phase_hpc_v2.sbatch
```

## Using The Pretrained Model

The package exposes a high-level loader that automatically uses the bundled pretrained checkpoint.

Minimal example:

```python
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from puicl import load_pretrained_model

torch.manual_seed(0)
device = "cpu"  # use "cuda" here if you want GPU inference
model = load_pretrained_model(device=device)

# Synthetic PU task:
# - labeled positives are sampled from one Gaussian
# - unlabeled rows are an explicit mixture of positive and negative Gaussians
train_size = 64
num_unlabeled_pos = 128
num_unlabeled_neg = 128
num_features = 12

positive_mean = torch.full((num_features,), 1.0, device=device)
negative_mean = torch.full((num_features,), -1.0, device=device)

labeled_pos = torch.randn(train_size, num_features, device=device) + positive_mean
unlabeled_pos = torch.randn(num_unlabeled_pos, num_features, device=device) + positive_mean
unlabeled_neg = torch.randn(num_unlabeled_neg, num_features, device=device) + negative_mean

X_unlabeled = torch.cat([unlabeled_pos, unlabeled_neg], dim=0)
y_true = torch.cat(
    [
        torch.zeros(num_unlabeled_pos, dtype=torch.long, device=device),
        torch.ones(num_unlabeled_neg, dtype=torch.long, device=device),
    ],
    dim=0,
)

# Shuffle so the unlabeled portion is a real mixture of the two Gaussians.
perm = torch.randperm(X_unlabeled.shape[0], device=device)
X_unlabeled = X_unlabeled[perm]
y_true = y_true[perm]

outlier_prob = model.score_unlabeled(labeled_pos, X_unlabeled)
y_pred = (outlier_prob >= 0.5).to(torch.long)

accuracy = accuracy_score(y_true.numpy(), y_pred.numpy())
auc = roc_auc_score(y_true.numpy(), outlier_prob.numpy())
cm = confusion_matrix(y_true.numpy(), y_pred.numpy())

print(f"num_unlabeled={len(y_true)}")
print(f"accuracy={accuracy:.4f}")
print(f"auc={auc:.4f}")
print(cm)
```

Example output:

```text
num_unlabeled=256
accuracy=0.9648
auc=1.0000
[[119   9]
 [  0 128]]
```

Input/output conventions:
- `load_pretrained_model()` loads the packaged checkpoint and returns a `PUICLModel` wrapper
- `score_unlabeled(labeled_positive_features, unlabeled_features)` returns the outlier probability for each unlabeled row
- `predict_logits(...)`, `predict_proba(...)`, and `predict_labels(...)` are available when you want direct control over the full PU task tensor
- class `1` corresponds to the outlier score used in evaluation

## Evaluation

The evaluator is standalone and embeds the benchmark protocol directly:
- same benchmark datasets
- same feature preprocessing
- same conversion from binary classification datasets to PU tasks
- same evaluation metrics

The evaluator runs the checkpoint in `pretrained_model/latest.pt` by default.

Important arguments:
- `--checkpoint`: path to the checkpoint to evaluate. Defaults to `pretrained_model/latest.pt`.
- `--output-dir`: directory where evaluation runs are saved. Each run creates `eval_<timestamp>/`.
- `--cache-dir`: directory for cached UCI downloads and parsed files. Defaults to `.cache/` inside the repo.
- `--device`: execution device. Use `auto`, `cpu`, `cuda`, or `mps`.
- `--allow-uci-download` / `--no-uci-download`: enable or disable downloading benchmark datasets that are not already cached.
- `--n-replicates`: number of independent PU tasks sampled per dataset.
- `--max-attempts-per-dataset`: maximum number of tries used to obtain valid PU tasks for a dataset. This matters when a dataset cannot always satisfy the requested PU constraints.
- `--global-seed`: random seed controlling dataset-level reproducibility.
- `--max-categorical-classes`: cap used when encoding categorical features after preprocessing.

PU-conversion arguments:
- `--max-positive-size`: maximum number of examples from the chosen positive class to use when constructing one PU task.
- `--unlabeled-positive-ratio` and `--labeled-positive-ratio`: define the split of selected positives between the unlabeled pool and the labeled prefix. For example, `2` and `1` means two-thirds of the selected positives go to the unlabeled pool and one-third remain labeled.
- `--outlier-rate`: target fraction of negatives inside the unlabeled pool. For example, `0.13` means the unlabeled set is built to contain about `13%` outliers and `87%` unlabeled positives.

What one replicate means:
- choose one of the two original class labels as the positive class
- sample up to `max_positive_size` positives from that class
- split those positives into labeled and unlabeled parts using the requested ratio
- add negatives into the unlabeled pool to match `outlier_rate`
- evaluate the model on that resulting PU task

Example with all major controls shown:

```bash
python evaluate_pretrained_model.py \
  --checkpoint pretrained_model/latest.pt \
  --device auto \
  --n-replicates 10 \
  --max-positive-size 900 \
  --unlabeled-positive-ratio 2 \
  --labeled-positive-ratio 1 \
  --outlier-rate 0.13 \
  --global-seed 42
```

Evaluation outputs are written to:

```text
evaluation_outputs/eval_<timestamp>/
```

and include:
- summary metrics
- per-replicate metrics
- dataset feature profiles
- realized PU composition summaries
- per-dataset feature metadata

## Dependencies

The code expects a Python environment with at least:
- `torch`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `xlrd`

Some evaluation datasets are downloaded from the UCI repository on first use and cached under `.cache/`.

## Notes

- Existing evaluation outputs in `evaluation_outputs/` are included as example benchmark runs with different PU settings.
