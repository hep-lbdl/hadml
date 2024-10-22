# Machine Learning For Hadronization

The structure of the repository is organised in the same way as in the 
[Lighting Template](https://github.com/ashleve/lightning-hydra-template) repository.
The original `README.md` is moved to [`tests/README.md`](tests/README.md).

# Quickstart Guide

*Last tested: 20.10.2024*


In this short guide we will prepare an environment and run some experiments. If you are a NERSC user,
please follow the ***"Quickstart Guide for NERSC users"*** as it contains additional steps aiming
to help you in running the model in a right way. Otherwise, follow the steps below and make sure you
machine fulfills the following requirements:

* Requirements: Python 3.9, Conda;
* Operating System: Linux;
* Storage: at least 50 GB;
* Memory: at least 30 GB.

```bash
# Cloning the project repository
$ git clone https://github.com/remilvus/hadml.git
$ cd hadml

# Creating a Conda virtual environment
$ conda create -n herwig python=3.9
$ conda activate herwig

# Installing packages via Conda: PyTorch (e.g. v12.4 with CUDA support) according to the instructions: https://pytorch.org/get-started/
(herwig) $ conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Installing other libraries
(herwig) $ conda install pyg -c pyg 
(herwig) $ conda install -c conda-forge wandb  

# Installing additional libraries with pip
(herwig) $ pip install --no-cache-dir pip==24.0
(herwig) $ pip install --no-cache-dir -r requirements.txt

# Downloading data from Zenodo (around 21 GB)
(herwig) $ wget "https://zenodo.org/records/10246934/files/allHadrons_with_quark.npz"

# Unpacking the NumPy archive, saving files as NPY files in data/Herwig/cache/ and getting rid of the archive
(herwig) $ python data/unpack_npz.py --filename "allHadrons_with_quark.npz"
(herwig) $ rm allHadrons_with_quark.npz

# Trying to run an experiment:
# Notice #1: "datamodule.core_dataset.frac_data_used" limits the amount of data used for training;
# Notice #2: add "trainer.accelerator=cpu" if you are testing the code on CPUs.
(herwig) $ python hadml/train.py experiment=herwig_all_hadron datamodule.core_dataset.cache_dir=data/Herwig/cache/ datamodule.core_dataset.frac_data_used=0.05 

# Deactivating the virtual environment
(herwig) $ conda deactivate
```

# Quickstart Guide for NERSC users


*Last tested: 14.10.2024*

---

The following instructions are different from what we described above. The steps include actions 
which are necessary for NERSC users who are going to run the code on Perlmutter (including loading
modules, allocating nodes etc). Replace `<username>`, `<project_number>` (e.g. m1234) and 
`<account_name>` (e.g m1234 or m1234_g) with appropriate values.

> NB you need to use a login node to be able to write in Global Common and install packages. 

```bash
# Creating a virtual environment
$ cd /global/common/software/<project_number>
$ mkdir <username>
$ module load conda
$ conda create -p /global/common/software/<project_number>/<username>/conda/herwig python=3.9

# Installing packages via Conda: PyTorch (e.g. v12.4 with CUDA support) according to the instructions: https://pytorch.org/get-started/
$ conda activate /global/common/software/<project_number>/<username>/conda/herwig
(herwig) $ conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Installing other libraries
(herwig) $ conda install pyg -c pyg 
(herwig) $ conda install -c conda-forge wandb
```

> NB the Common File System (CFS) provides you with medium performance and permanent storage. 
The Scratch file system takes advantage of the best performance but can only store files for a 
limited period of time (7 days).

```bash
(herwig) $ cd $CFS/<project_number>
(herwig) $ mkdir <username>
(herwig) $ cd <username>
```

or

```bash
(herwig) $ cd $SCRATCH
```
---

```bash
# Cloning the project repository:
(herwig) $ git clone https://github.com/remilvus/hadml.git
(herwig) $ cd hadml

# Installing packages with pip    
(herwig) $ pip install --no-cache-dir pip==24.0
(herwig) $ pip install --no-cache-dir -r requirements.txt

# Downloading data from Zenodo
(herwig) $ wget "https://zenodo.org/records/10246934/files/allHadrons_with_quark.npz"

# Unpacking the NumPy archive, saving files as NPY files in data/Herwig/cache/ and getting rid of the archive
(herwig) $ python data/unpack_npz.py --filename "allHadrons_with_quark.npz"
(herwig) $ rm allHadrons_with_quark.npz
(herwig) $ conda deactivate

# Allocating an interactive GPU node (you may specify time on your own)
$ salloc --nodes 1 --qos interactive --time 00:10:00 --constraint gpu --gpus 4 --account <account_name, e.g. m1234 or m1234_g>

# Activating the prepared Conda environment on the allocated node
$ module load conda
$ conda activate /global/common/software/<project_number>/<username>/conda/herwig

# Trying to run an experiment:
# Notice #1: "datamodule.core_dataset.frac_data_used" limits the amound of data used;
# Notice #2: add "trainer.accelerator=cpu" if you are testing the code on CPUs.
(herwig) $ python hadml/train.py experiment=herwig_all_hadron datamodule.core_dataset.cache_dir=data/Herwig/cache/ datamodule.core_dataset.frac_data_used=0.05 
```

# Experiments
After having prepared the environment, feel free to run different experiments:

```bash
# Running the training process for cluster-level generation
(herwig) $ python hadml/train.py experiment=herwig_all_hadron datamodule.core_dataset.cache_dir=data/Herwig/cache/ 

# Running the training process for event-level generation
(herwig) python hadml/train.py experiment=herwig_event datamodule.core_dataset.cache_dir=data/Herwig/cache/ 
```

You may also start training for the event-level generation with fitting to the nominal or variation
Herwig sample. You need to download data samples from https://doi.org/10.5281/zenodo.7958362 and
then place it at `data/Herwig/raw`: 

```bash
# Downloading data
(herwig) $ wget "https://zenodo.org/records/7958362/files/ClusterTo2Pi0_nominal.dat"
(herwig) $ wget "https://zenodo.org/records/7958362/files/ClusterTo2Pi0_variation.dat"
(herwig) $ mv ClusterTo2Pi0_nominal.dat data/Herwig/raw
(herwig) $ mv ClusterTo2Pi0_variation.dat data/Herwig/raw

# Running experiments
(herwig) $ python hadml/train.py experiment=herwig_event_nominal
(herwig) $ python hadml/train.py experiment=herwig_event_variation
```

# Crash Course On Training
This nice code structure is based on the [Pytorch Lightning](https://www.pytorchlightning.ai/) and
[Hydra](https://hydra.cc/docs/intro/). It is created from the 
[Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template). 
PyTorch lightning saves us from writing many boilerplates. Hydra provides powerful configuration 
management and allows us to easily switch between different configurations.
However, great flexibility comes with many options/configurations and it can be overwhelming at first.

`configs` contains a hierarchical configurations for different parts of the code.
Some of them are used to initalise the model or to configure the training.

Models are defined in [`hadml/models`](hadml/models) and the data modules
are defined in [`hadml/datamodules`](hadml/datamodules).

First create a local `.env` file with the following content:
```bash
HerwigData="/path/to/lightning-hydra-template/data"
```
The environment variable `HerwigData` is used to locate the data files.

## Training "Cluster Decayer" With Particle Type Labels

*Last tested: not tested yet.*

---

To perform training, run the following command:
```bash
$ python hadml/train.py experiment=herwig_all_hadron
```
This will train a model using `configs/experiment/herwig_all_hadron.yaml`.
Results and logs will be saved in `logs/herwigAllhadron/runs` as defined by `hydra.run.dir` in 
`configs/hydra/default.yaml`. You can change the directory by changing the `task_name` in 
`configs/experiment/herwig_all_hadron.yaml`.

### Training with a logger
It is optional to add a logger. For example, you can monitor the training 
performance using the [Weights & Biases](https://wandb.ai/site) logger by running:

```bash
$ python hadml/train.py experiment=herwig_all_hadron logger=wandb logger.wandb.project=<project_name>
```
You are supposed to replace <project_name> with a project created in advance in you W&B profile.
Read more: [W&B Quickstart](https://docs.wandb.ai/quickstart).

### Training with different training options
[Pytorch Trainer](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html) provides a 
lot of options for training. For example, you can start training with gradient clip by running:
```bash
$ python hadml/train.py experiment=herwig_all_hadron logger=wandb +trainer.gradient_clip_val=0.5
```

## Training "Cluster Decayer" With Event Level Information

*Last tested: not tested yet.*

---

Similarly, the training can be performed by running:
```bash
$ python hadml/train.py experiment=herwig_event logger=wandb
```
---
Last update: 14.10.2024