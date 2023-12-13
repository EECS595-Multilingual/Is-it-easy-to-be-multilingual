#!/bin/bash

#SBATCH --partition=spgpu
#SBATCH --time=00-00:30:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32GB
#SBATCH --account=eecs595f23_class

# set up job
module load python cuda
pushd /home/kpyu/great-lakes-tutorial
source venv/bin/activate

# run job
python3 PANX_mBERT_script.py

