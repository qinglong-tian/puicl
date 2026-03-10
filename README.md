# PU_ICL_Code

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
- `evaluate_pretrained_model_reference.ipynb`: reference notebook used as the source of truth for benchmark datasets, PU conversion, and metrics
- `evaluation_outputs/`: saved benchmark runs
- `run_pretrain_two_phase_hpc_v2.sbatch`: two-phase HPC training launcher

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

## Evaluation

The evaluator is designed to preserve the benchmark protocol defined in the reference notebook:
- same benchmark datasets
- same feature preprocessing
- same conversion from binary classification datasets to PU tasks
- same evaluation metrics

The evaluator runs the checkpoint in `pretrained_model/latest.pt` by default.

Example:

```bash
python evaluate_pretrained_model.py
```

PU task construction can be controlled from the command line. Example:

```bash
python evaluate_pretrained_model.py \
  --max-positive-size 900 \
  --unlabeled-positive-ratio 2 \
  --labeled-positive-ratio 1 \
  --outlier-rate 0.13
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

- `evaluate_pretrained_model.py` uses `evaluate_pretrained_model_reference.ipynb` as a reference implementation for the benchmark protocol, so the repository remains aligned with the original notebook logic.
- Existing evaluation outputs in `evaluation_outputs/` are included as example benchmark runs with different PU settings.
