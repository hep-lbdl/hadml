# Machine Learning for Hadronization

This repository contains deep generative models for simulating hadron production in high-energy physics, featuring **Set Transformers** (**cGANs**).

The project structure follows the [Lightning Hydra Template](https://github.com/ashleve/lightning-hydra-template).

---

## Quickstart Guide

*Last tested: 24 July 2026 (macOS 26.5.2 / Apple Silicon M5)*

### Requirements
* **Python:** 3.14
* **Package Manager:** Conda & Pip

---

### Environment Setup

```bash
# Clone the repository
git clone [https://github.com/hep-lbdl/hadml](https://github.com/hep-lbdl/hadml)
cd hadml

# Create and activate Conda environment
conda create -n herwig python=3.14
conda activate herwig

# Install core dependencies via pip
python -m pip install torch torchvision
python -m pip install lightning hydra-core hydra-colorlog pyrootutils rich
python -m pip install torch_geometric pandas scipy matplotlib scikit-learn wandb POT


```

### Training Configuration & Options
Configurations are managed dynamically using Hydra and orchestrated via PyTorch Lightning.
• Config directory: `configs/` contains hierarchical configurations for datasets, models, trainers, and loggers.
• Model code: Located in `hadml/models/`.
• Data modules: Located in `hadml/datamodules/`.

### Experiment Monitoring (Weights & Biases)
To log experiments to Weights & Biases:
```bash
python hadml/train.py experiment=herwig_event_multihadron logger=wandb logger.wandb.project=<your_wandb_project>
```
