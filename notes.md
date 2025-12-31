### Setup code

module purge || true
unset LD_PRELOAD || true
export XALT_EXECUTABLE_TRACKING=0
export XALT_RUNNABLE=0
unset LD_LIBRARY_PATH
source ~/miniforge3/etc/profile.d/conda.sh
module load CUDA/12.9.1
export CONDA_NO_PLUGINS=1
conda activate ./trm

### basic environment setup
```bash
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129 # install torch based on your cuda version
pip install -r requirements.txt # install requirements
pip uninstall -y adam-atan2
pip install adam-atan2-pytorch
wandb login 890e92a2bb020c0784046ca55232006b6aa2307f # login if you want the logger to sync results to your Weights & Biases (https://wandb.ai/)
```

### test env (debug shell)
srun --pty --time=4:00:00 --gres=gpu:h100:1 --mem=80G $SHELL

### run test
python -u pretrain.py \
    arch=trm \
    data_paths="[data/sudoku-extreme-1k-aug-1000]" \
    evaluators="[]" \
    epochs=50000 eval_interval=5000 \
    lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
    arch.mlp_t=True arch.pos_encodings=none \
    arch.L_layers=2 \
    arch.H_cycles=3 arch.L_cycles=6 \
    +run_name=trm_r_mlp-sudoku_extreme ema=True

### codex commmand

codex --dangerously-bypass-approvals-and-sandbox

### check allocation status
sinfo -N -O "NodeList,CPUsState,Memory,FreeMem,Gres,GresUsed"   | awk 'NR==1{print; next} {print $1, $2, $3/1024, $4/1024, $5, $6}'

### Evaluation
python evaluate.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  +load_checkpoint=checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/trm_r_mlp-sudoku_extreme/step_65100 \
  +project_name=Sudoku-extreme-1k-aug-1000-ACT-torch \
  +run_name=eval_trm_mlp
