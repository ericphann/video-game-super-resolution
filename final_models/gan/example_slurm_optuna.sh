# Update your file to match your hpc specs.

#!/bin/bash
#SBATCH --job-name=srgan_optuna
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=srgan_optuna_output.txt

# Load modules if needed
module load python/3.12.3
module load tensorflow/2.16.1-cuda12.5

# Set environment variable
export TF_GPU_ALLOCATOR=cuda_malloc_async

# Install required Python packages
pip install --user optuna matplotlib pandas numpy

# Logging
echo "Starting Optuna tuning at: $(date)"
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"

# Run Optuna tuning script
export DATASET_PATH="/path/video-game-sr" # update path accordingly

echo "Running Optuna Hyperparameter tuning..."
python optuna_tuner_full.py
echo "Optuna tuning completed at: $(date)"
