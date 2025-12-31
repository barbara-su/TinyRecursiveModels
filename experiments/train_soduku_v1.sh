#!/bin/bash
#SBATCH -A ak85
#SBATCH -p commons
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=23:00:00
#SBATCH --job-name=trm_mlp
#SBATCH --output=/scratch/bs82/TinyRecursiveModels/logs/soduku-%x-%j.log
#SBATCH --error=/scratch/bs82/TinyRecursiveModels/logs/soduku-%x-%j.err

cd /scratch/bs82/TinyRecursiveModels

module purge || true
unset LD_PRELOAD || true
export XALT_EXECUTABLE_TRACKING=0
export XALT_RUNNABLE=0
unset LD_LIBRARY_PATH
source ~/miniforge3/etc/profile.d/conda.sh
module load CUDA/12.9.1
export CONDA_NO_PLUGINS=1
conda activate ./trm

wandb login 890e92a2bb020c0784046ca55232006b6aa2307f

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting TRM training suite"

# original trm mlp
torchrun --nproc-per-node=4 pretrain.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=50000 \
  eval_interval=5000 \
  lr=1e-4 puzzle_emb_lr=1e-4 \
  weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.mlp_t=True arch.pos_encodings=none \
  arch.H_cycles=3 arch.L_cycles=6 \
  +run_name=trm_r_mlp-sudoku_extreme ema=True

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training suite completed"