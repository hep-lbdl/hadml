#!/bin/bash

## Specifying the account and queue
#SBATCH --account=<group_id>_g
#SBATCH --qos=<debug | regular>

## Running on multiple GPUs on the same node via multiple processes (the DDP strategy)
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4

## Specifying time, logs and the job name
#SBATCH --time=<00:20:00 | 24:00:00>
#SBATCH --job-name=pions
#SBATCH --output=slurm-%x_%j.out
#SBATCH --output=slurm-%x_%j.err

## Loading conda
module load conda

## Activating the virtual evironment
conda activate /global/common/software/<group_id>/<user_id>/conda/<virtual_evironment>

## Debugging flag
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

## Running the program via srun (it is needed for multiple GPUs signal communication)
srun python hadml/train.py experiment=herwig_event_multihadron logger=wandb