# Revealing Latent Domains via Unsupervised Symmetry Discovery

Official code release for the paper:

> **Revealing Latent Domains via Unsupervised Symmetry Discovery**  
> Onur Efe, Arkadas Özakın  
> Preprint, 2026  
> [arXiv link coming soon]

---

## Overview

This repository contains the implementation of the **Group-Equivariant Lifting Layer** — a neural network layer that simultaneously discovers the latent symmetry group and locality structure of data without supervision.

Given observations drawn from an unknown group action on an unknown signal class, the model jointly learns:
- A **lifting map** $L$ that embeds input data into a structured representation
- The **generator** of the symmetry group that best explains the data's statistical structure

The learning objective minimises **total correlation** and maximises **uniformity** of the lifted representation across group positions.

---

## Quick start

```bash
git clone <this-repo>
cd LatentDomainRecovery-release
pip install -r requirements.txt
jupyter notebook notebooks/quickstart.ipynb
```

---

## Repository structure

```
LatentDomainRecovery-release/
├── core/                          # Main library
│   ├── lifting_layer.py           # LiftingLayer: group-equivariant spectral lifting
│   ├── models.py                  # Full model (lifting + uniformity/TC losses)
│   ├── uniformity_estimator.py    # Trainable CNN uniformity estimator (JS divergence)
│   ├── probability_estimator.py   # Kernel density TC estimator
│   ├── synthetic_data_generator.py      # 1-D and 2-D data generators
│   ├── neural_data_preprocessors.py     # Allen Brain Neuropixel preprocessing
│   ├── train_utils.py             # Optimizers, LR scheduling, training entry point
│   └── training_loop.py           # Multi-GPU training loop
│
├── baselines/                     # Competing methods
│   ├── glasso-waveform/           # Graphical LASSO on 1-D waveforms
│   ├── glasso-mnist/              # Graphical LASSO on MNIST crops
│   ├── tica-gsn/                  # TICA on GSN waveforms
│   ├── tica-ising/                # TICA on Ising model samples
│   ├── tica-mnist/                # TICA on MNIST crops
│   ├── lgan-gsn/                  # Latent GAN on GSN waveforms
│   ├── manifold-mnist/            # UMAP / Isomap on MNIST crops
│   └── manifold-neuropixel/       # UMAP / Isomap on Neuropixel spikes
│
├── run-configs/                   # JSON experiment configurations
│   ├── waveform-experiments.json  # 1-D waveform experiments (Gaussian & Legendre, 5 seeds each)
│   ├── mnist-experiments.json     # 2-D MNIST translation experiments
│   └── neuropixel-experiments.json  # Allen Brain Neuropixel experiments
│
├── notebooks/
│   ├── quickstart.ipynb           # Minimal end-to-end demo (~5 min)
│   ├── train.ipynb                # Full training from a JSON config
│   ├── evaluate.ipynb             # Load weights, visualise generators & lifting map
│   └── baselines.ipynb            # Run and compare baselines
│
├── neuropixel-session-download.py # Allen Brain dataset download script
└── experiment_runner.py           # CLI multi-GPU experiment launcher
```

---

## Notebooks

| Notebook | Description | Time |
|---|---|---|
| `quickstart.ipynb` | Minimal demo: train on 1-D Gaussians, inspect learned generators | ~5 min |
| `train.ipynb` | Full training run from any `run-configs/*.json` | hours |
| `evaluate.ipynb` | Load a checkpoint, visualise lifting map, measure circulant alignment | seconds |
| `baselines.ipynb` | Run TICA & GLASSO; load and plot pre-computed baseline results | ~10 min |

---

## Running full experiments (CLI)

The `experiment_runner.py` script reads a JSON config and launches all experiments in parallel across GPUs:

```bash
python experiment_runner.py --config run-configs/waveform-experiments.json
```

Each experiment writes logs and periodic weight checkpoints under `experiments/<EXP_NAME>/`.

---

## Dependencies

```
tensorflow~=2.14.0
numpy<2
scipy>=1.10.0
matplotlib>=3.7.0
jupyter>=1.0.0
gitpython>=3.1.0
allensdk>=2.15.0    # only needed for Neuropixel experiments
umap-learn>=0.5.3   # only needed for manifold baselines
scikit-learn>=1.2.0 # only needed for GLASSO baseline
```

Install with:
```bash
pip install -r requirements.txt
```

For Apple Silicon (M1/M2/M3):
```bash
pip install tensorflow-macos tensorflow-metal
```

---

## Neuropixel data

The Neuropixel experiments use electrophysiology recordings from the [Allen Brain Institute](https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html).

To download the required session:
```bash
python neuropixel-session-download.py
```

Downloaded data is cached locally and is referenced by session ID in `run-configs/neuropixel-experiments.json`.

---

## Citation

```bibtex
@misc{efe2026revealing,
  title         = {Revealing Latent Domains via Unsupervised Symmetry Discovery},
  author        = {Onur Efe and Arkadas {\"O}zak{\i}n},
  year          = {2026},
  note          = {Preprint}
}
```
