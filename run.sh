#!/bin/bash
#SBATCH --job-name=my_job           # Job name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --tasks-per-node=1           # Number of tasks (usually 1 for GPU jobs)
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --gres=gpu:2                 # Number of GPUs per node
#SBATCH --partition=gpu              # GPU partition
#SBATCH --mem=32GB                   # Memory per node

# Load required modules and activate Conda environment
module load cuda/11.0
module load python/3.8
conda activate HCN

# Move to the directory where your Python script is located
cd /home/safar/HCN/src/huggingface_multiwoz

# Run the Python script
python main.py
