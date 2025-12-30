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
wandb login YOUR-LOGIN # login if you want the logger to sync results to your Weights & Biases (https://wandb.ai/)
```


### test env (debug shell)
srun --pty --time=4:00:00 --gres=gpu:h100:1 --mem=80G $SHELL


run test
python pretrain.py arch=trm 'data_paths=[data/sudoku-extreme-1k-aug-1000]' 'evaluators=[]' epochs=1 eval_interval=1 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 arch.H_cycles=3 arch.L_cycles=6 +boundary_only_correction=true +boundary_only_eps=1e-6 +kstep_correction=1 +checkpoint_path=checkpoints/debug_bound_k1 ema=True

python pretrain.py \
  arch=trm \
  arch.mlp_t=True \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=50000 \
  eval_interval=5000 \
  lr=1e-4 puzzle_emb_lr=1e-4 \
  weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.H_cycles=3 arch.L_cycles=6 \
  +boundary_only_correction=true \
  +boundary_only_eps=1e-8 \
  +kstep_correction=1 \
  +run_name=trm_mlp_test-sudoku_extreme ema=True \
  +checkpoint_path="checkpoints/test"

codex --dangerously-bypass-approvals-and-sandbox