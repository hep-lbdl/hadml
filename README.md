# Machine Learning For Hadronization

The structure of the repository is organised in the same way as in the 
[Lighting Template](https://github.com/ashleve/lightning-hydra-template) repository.

# Quickstart Guide

*Last tested: 20.10.2024*


In this short guide we will prepare an environment and run some experiments. If you are a NERSC user,
please follow the ***"Quickstart Guide for NERSC users"*** as it contains additional steps aiming
to help you in running the model in a right way. Otherwise, follow the steps below and make sure you
machine fulfills the following requirements:

* Requirements: Python 3.9, Conda;
* Operating System: Linux;

```bash
# Cloning the project repository
$ git clone https://github.com/hep-lbdl/hadml
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

# Running an experiment
(herwig) $ python hadml/train.py experiment=herwig_event_multihadron

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
(herwig) $ git clone https://github.com/hep-lbdl/hadml
(herwig) $ cd hadml

# Installing packages with pip    
(herwig) $ pip install --no-cache-dir pip==24.0
(herwig) $ pip install --no-cache-dir -r requirements.txt

# Allocating an interactive GPU node (you may specify time on your own)
$ salloc --nodes 1 --qos interactive --time 00:10:00 --constraint gpu --gpus 4 --account <account_name, e.g. m1234 or m1234_g>

# Activating the prepared Conda environment on the allocated node
$ module load conda
$ conda activate /global/common/software/<project_number>/<username>/conda/herwig

# Running an experiment
(herwig) $ python hadml/train.py experiment=herwig_event_multihadron
```

# Code Structure
This nice code structure is based on the [Pytorch Lightning](https://www.pytorchlightning.ai/) and
[Hydra](https://hydra.cc/docs/intro/). It is created from the 
[Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template). 
PyTorch lightning saves us from writing many boilerplates. Hydra provides powerful configuration 
management and allows us to easily switch between different configurations.

`configs` contains a hierarchical configurations for different parts of the code.
Some of them are used to initalise the model or to configure the training.

Models are defined in [`hadml/models`](hadml/models) and the data modules
are defined in [`hadml/datamodules`](hadml/datamodules).

First create a local `.env` file with the following content:
```bash
HerwigData="/path/to/lightning-hydra-template/data"
```
The environment variable `HerwigData` is used to locate the data files.

### Training with a logger
It is optional to add a logger. For example, you can monitor the training 
performance using the [Weights & Biases](https://wandb.ai/site) logger by running:

```bash
$ python hadml/train.py experiment=herwig_event_multihadron logger=wandb
```
You are supposed to replace <project_name> with a project created in advance in you W&B profile.
Read more: [W&B Quickstart](https://docs.wandb.ai/quickstart).

---

Last update: 24.07.2026