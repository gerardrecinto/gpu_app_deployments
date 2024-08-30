#!/bin/bash
#SBATCH --job-name=pytorch_gpu_job
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH --time=02:00:00         # Max time for job
#SBATCH --partition=gpu         # Submit to GPU partition

module load anaconda3/2021.11  # Load Anaconda or Python module
source activate pytorch_env    # Activate your PyTorch environment

srun python train.py           # Run the PyTorch training script

