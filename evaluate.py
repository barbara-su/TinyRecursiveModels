# evaluate.py
from __future__ import annotations

from typing import Optional, Any, Sequence, List, Dict
from dataclasses import dataclass
import os
import re
import json
import math
import yaml
import importlib
import importlib.util
import sys

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import hydra
import pydantic
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

from adam_atan2_pytorch import AdamAtan2

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams (kept for config compatibility, even in eval-only)
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0
    eval_save_outputs: List[str] = []

    ema: bool = False
    ema_rate: float = 0.999
    freeze_weights: bool = False


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def _parse_step_from_path(path: Optional[str]) -> int:
    if not path:
        return 0
    base = os.path.basename(path)
    m = re.search(r"step_(\d+)", base)
    if m:
        return int(m.group(1))
    return 0


def create_dataloader(
    config: PretrainConfig,
    split: str,
    rank: int,
    world_size: int,
    **kwargs,
):
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed,
            dataset_paths=(
                config.data_paths_test
                if (len(config.data_paths_test) > 0 and split == "test")
                else config.data_paths
            ),
            rank=rank,
            num_replicas=world_size,
            **kwargs,
        ),
        split=split,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True,
    )
    return dataloader, dataset.metadata


def _ensure_module_from_checkpoint(identifier: str, checkpoint_path: Optional[str]) -> None:
    module_path, _ = identifier.split("@")
    module_name = f"models.{module_path}"

    candidate_path = None
    if checkpoint_path is not None:
        checkpoint_dir = checkpoint_path
        if os.path.isfile(checkpoint_dir):
            checkpoint_dir = os.path.dirname(checkpoint_dir)

        module_basename = module_path.replace("/", ".").split(".")[-1]
        module_filename = f"{module_basename}.py"
        maybe_candidate = os.path.join(checkpoint_dir, module_filename)
        if os.path.exists(maybe_candidate):
            candidate_path = maybe_candidate

    if candidate_path is not None:
        spec = importlib.util.spec_from_file_location(module_name, candidate_path)
        if spec is None or spec.loader is None:
            raise ModuleNotFoundError(f"Cannot load module {module_name} from {candidate_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return

    importlib.import_module(module_name)


def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is None:
        return
    print(f"Loading checkpoint {config.load_checkpoint}")

    state_dict = torch.load(config.load_checkpoint, map_location="cuda")

    # Resize and reset puzzle emb if needed
    puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
    expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore[attr-defined]
    if puzzle_emb_name in state_dict:
        puzzle_emb = state_dict[puzzle_emb_name]
        if puzzle_emb.shape != expected_shape:
            print(
                f"Resetting puzzle embedding. Found {puzzle_emb.shape}, expected {expected_shape}"
            )
            state_dict[puzzle_emb_name] = (
                torch.mean(puzzle_emb, dim=0, keepdim=True)
                .expand(expected_shape)
                .contiguous()
            )

    model.load_state_dict(state_dict, assign=True)


def create_model(config: PretrainConfig, metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore[union-attr]
        batch_size=config.global_batch_size // world_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False,
    )

    _ensure_module_from_checkpoint(config.arch.name, config.load_checkpoint)
    _ensure_module_from_checkpoint(config.arch.loss.name, config.load_checkpoint)

    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore[union-attr]

        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore[assignment]

        if rank == 0:
            load_checkpoint(model, config)

        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers are not used in eval-only, but we keep the same shape for TrainState.
    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [
            AdamAtan2(
                model.parameters(),
                lr=1e-8,
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2),
            )
        ]
        optimizer_lrs = [config.lr]
    elif config.freeze_weights:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore[attr-defined]
                lr=1e-8,
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size,
            )
        ]
        optimizer_lrs = [config.puzzle_emb_lr]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore[attr-defined]
                lr=1e-8,
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size,
            ),
            AdamAtan2(
                model.parameters(),
                lr=1e-8,
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2),
            ),
        ]
        optimizer_lrs = [config.puzzle_emb_lr, config.lr]

    return model, optimizers, optimizer_lrs


def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths = config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            evaluator = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )  # type: ignore[misc]
            evaluators.append(evaluator)
    return evaluators


def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
):
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        save_preds: Dict[str, List[torch.Tensor]] = {}

        metric_keys: List[str] = []
        metric_values: Optional[torch.Tensor] = None

        processed_batches = 0
        for set_name, batch, _global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")

            batch = {k: v.cuda() for k, v in batch.items()}

            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore[attr-defined]

            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1
                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            set_id = set_ids[set_name]
            if metric_values is None:
                metric_keys = sorted(metrics.keys())
                metric_values = torch.zeros(
                    (len(set_ids), len(metric_keys)), dtype=torch.float32, device="cuda"
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del carry, loss, preds, batch, all_finish, metrics

        # Save prediction tensors per rank, if requested
        if config.checkpoint_path is not None and len(save_preds) > 0:
            os.makedirs(config.checkpoint_path, exist_ok=True)
            save_preds_cat = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}
            torch.save(
                save_preds_cat,
                os.path.join(config.checkpoint_path, f"eval_step_{train_state.step}_preds.rank{rank}.pt"),
            )
            del save_preds_cat

        # Reduce metrics to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                arr = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: float(arr[set_id, metric_id])
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess: divide by count
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count", 1.0)
                    count = max(count, 1.0)
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        for evaluator in evaluators:
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            metrics = evaluator.result(
                evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group
            )
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}
                reduced_metrics.update(metrics)

    return reduced_metrics


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects: List[Optional[PretrainConfig]] = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore[arg-type]

        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
        if config.run_name is None:
            config.run_name = "eval"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    assert objects[0] is not None
    return objects[0]


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def main(hydra_config: DictConfig):
    # Make relative paths behave like your training script expects.
    os.chdir(get_original_cwd())

    rank = 0
    world_size = 1
    cpu_group = None

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        cpu_group = dist.new_group(backend="gloo")
        assert dist.get_rank(cpu_group) == rank
        assert dist.get_world_size(cpu_group) == world_size

    config = load_synced_config(hydra_config, rank=rank, world_size=world_size)

    if config.load_checkpoint is None:
        raise ValueError("You must set load_checkpoint=... to evaluate a checkpoint.")

    torch.random.manual_seed(config.seed + rank)

    # Build eval loader
    try:
        eval_loader, eval_metadata = create_dataloader(
            config,
            "test",
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size,
            rank=rank,
            world_size=world_size,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to build test dataloader. "
            "Set data_paths_test=[...] or ensure your dataset contains a test split."
        ) from e

    # Model and evaluators
    model, optimizers, optimizer_lrs = create_model(config, eval_metadata, rank=rank, world_size=world_size)

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except Exception:
        evaluators = []

    step = _parse_step_from_path(config.load_checkpoint)
    train_state = TrainState(
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
        step=step,
        total_steps=0,
    )

    train_state.model.eval()

    metrics = evaluate(
        config,
        train_state,
        eval_loader,
        eval_metadata,
        evaluators,
        rank=rank,
        world_size=world_size,
        cpu_group=cpu_group,
    )

    # Save metrics on rank 0
    if rank == 0:
        print("\n=== Evaluation metrics ===")
        print(json.dumps(metrics, indent=2) if metrics is not None else "None")

        if config.checkpoint_path is not None:
            os.makedirs(config.checkpoint_path, exist_ok=True)
            out_path = os.path.join(config.checkpoint_path, f"eval_step_{train_state.step}.json")
            with open(out_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"\nSaved metrics to: {out_path}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
