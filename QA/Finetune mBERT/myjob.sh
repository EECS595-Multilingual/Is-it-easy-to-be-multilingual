#!/bin/bash

#SBATCH --partition=spgpu
#SBATCH --time=00-06:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64GB
#SBATCH --account=eecs595f23_class

# set up job
module load python cuda
pushd /home/kpyu/great-lakes-tutorial
source venv/bin/activate

# run job
python3 QA_script.py