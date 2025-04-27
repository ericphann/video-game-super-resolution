## Update the specs of your hpc cluster accordingly. These are placeholders
#!/bin/bash
#SBATCH --job-name=srgan_train
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=srgan_train_output.txt

# Load modules if needed
module load python/3.12.3
module load tensorflow/2.16.1-cuda12.5

# Install dependencies
pip install --user keras matplotlib numpy pandas

# Run SRGAN training
RUN_NAME="run_$(date +%m%d_%H%M)"
CHECKPOINT_DIR="/users/smicha17/checkpoints/$RUN_NAME"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p /path/figures ## update your path accordingly

echo "Script starting at $(date)"
echo "Running on host: $(hostname)"
echo "Checkpoint directory: $CHECKPOINT_DIR"

DATASET_PATH="/path/video-game-sr" ## update your path accordingly
echo "Using dataset at: $DATASET_PATH"

if [ -d "$DATASET_PATH" ]; then
    echo "Dataset found. Starting training..."

    python train_srgan.py \
        --use_best_params \
        --epochs=200 \
        --checkpoint_dir=$CHECKPOINT_DIR \
        --steps_per_epoch=5000

    if [ -f "optuna_trials/best_trial.txt" ]; then
        cp optuna_trials/best_trial.txt "$CHECKPOINT_DIR/best_params_used.txt"
        echo "Copied best_trial.txt to checkpoint directory."
    fi
else
    echo "Dataset not found at $DATASET_PATH. Please check the path."
    exit 1
fi

echo "Training complete at $(date)"
