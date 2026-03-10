# In-Context Learning for Positive-Unlabeled Classification and Outlier Detection

Code and experiment artifacts for PU in-context learning on tabular data.

This repository contains:
- the PU-adapted TabPFN-style model used in the paper
- synthetic-prior pretraining code and curriculum setup
- an evaluation pipeline on a fixed set of binary tabular benchmarks
- a pretrained checkpoint and example evaluation outputs

## Repository Layout

- `model.py`: main `NanoTabPFNPUModel`
- `train/`: pretraining config, trainer, schedules, and launch entry points
- `simplified_prior/`: synthetic data generator and curriculum sampler
- `data/`: padded-batch generation for PU pretraining
- `pretrained_model/latest.pt`: pretrained checkpoint
- `evaluate_pretrained_model.py`: benchmark evaluator for the pretrained checkpoint
- `evaluation_outputs/`: saved benchmark runs
- `run_pretrain_two_phase_hpc_v2.sbatch`: two-phase HPC training launcher

## Quick Start

Evaluate the bundled checkpoint:

```bash
python evaluate_pretrained_model.py
```

The evaluator is standalone and does not depend on any external notebook.

Run the same benchmark with custom PU conversion settings:

```bash
python evaluate_pretrained_model.py \
  --max-positive-size 900 \
  --unlabeled-positive-ratio 2 \
  --labeled-positive-ratio 1 \
  --outlier-rate 0.13
```

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

The provided Slurm launcher currently requests:
- `1` node
- `2 x H100` GPUs
- `12` CPU cores
- `32G` RAM

Example launch:

```bash
sbatch /path/to/PU_ICL_Code/run_pretrain_two_phase_hpc_v2.sbatch
```

## Using The Pretrained Model

The bundled checkpoint is stored at `pretrained_model/latest.pt`.

Minimal example:

```python
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from model import NanoTabPFNPUModel

torch.manual_seed(0)
device = "cpu"  # use "cuda" here if you want GPU inference
checkpoint_path = Path("pretrained_model/latest.pt")

payload = torch.load(checkpoint_path, map_location=device)
model_cfg = payload["config"]["model"]

model = NanoTabPFNPUModel(
    embedding_size=int(model_cfg["embedding_size"]),
    num_attention_heads=int(model_cfg["num_attention_heads"]),
    mlp_hidden_size=int(model_cfg["mlp_hidden_size"]),
    num_layers=int(model_cfg["num_layers"]),
    num_outputs=int(model_cfg["num_outputs"]),
).to(device)
model.load_state_dict(payload["model_state_dict"])
model.eval()

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

X_task = torch.cat([labeled_pos, X_unlabeled], dim=0)
y_train = torch.zeros(train_size, device=device)

with torch.no_grad():
    logits = model(
        (X_task.unsqueeze(0), y_train.unsqueeze(0)),
        train_test_split_index=train_size,
    ).squeeze(0)
    probs = torch.softmax(logits, dim=-1)
    outlier_prob = probs[:, 1]
    y_pred = torch.argmax(logits, dim=-1)

accuracy = accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
auc = roc_auc_score(y_true.cpu().numpy(), outlier_prob.cpu().numpy())
cm = confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy())

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
- `X_task` should be a float tensor of shape `[rows, features]`
- `y_train` contains labels only for the labeled prefix, with `0` for labeled positives/inliers
- the model returns logits only for rows after `train_test_split_index`
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
