# Machine Learning for Hadronization

The structure of the repository is organized as the same as the 
[Lighting Template](https://github.com/ashleve/lightning-hydra-template).
The original README.md is moved to [`tests/README.md`](tests/README.md).

# A crash course on training
This nice code structure is based on the [Pytorch Lightning](https://www.pytorchlightning.ai/) and [Hydra](https://hydra.cc/docs/intro/). It is created from the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template). 
So all our ML models will use the Pytorch. Hydra provides powerful configuration management and allows us to easily switch between different configurations.
However, great flexibility comes with many options/configurations and it can be overwhelming at first.

`configs` contains a hieratical configurations for different parts of the code.
Some of them are used to initalize the model or to configure the training.

Models are defined in [`hadml/models`](hadml/models) and the data modules
are defined in [`hadml/datamodules`](hadml/datamodules).

First create a local `.env` file with the following content:
```bash
HerwigData="/path/to/lightning-hydra-template/data"
```
The environment variable `HerwigData` is used to locate the data files.

## Training the "Cluster Decayer" with particle type labels
(not tested yet)

To perform a training, run the following command:
```bash
python hadml/train.py experiment=herwig_all_hadron
```
This will train a model using the `configs/experiment/herwig_all_hadron.yaml`.
Results and logs will be saved in the `logs/herwigAllhadron/runs` as defined by `hydra.run.dir` in the `configs/hydra/default.yaml`. You can change the directory by changing the `task_name` in the `configs/experiment/herwig_all_hadron.yaml`.

### Training with a logger
It is optional to add a logger. For example, you can monitor the training 
performance using the [Weights & Biases](https://wandb.ai/site) logger by running:
```bash
python hadml/train.py experiment=herwig_all_hadron logger=wandb
```

### Training with different training options
[Pytorch Trainer](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html) provides a lot of options for training. For example, you can run the training with gradient clip by running:
```bash
python hadml/train.py experiment=herwig_all_hadron logger=wandb +trainer.gradient_clip_val=0.5
```

## Training the "Cluster Decayer" with event level information
(not tested yet)

Similarly, the training can be performed by running:
```bash
python hadml/train.py experiment=herwig_event logger=wandb
```