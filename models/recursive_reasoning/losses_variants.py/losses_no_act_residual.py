from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        loss_type: str,
        fp_iters_weight: float = 0.0,
        fp_residual_weight: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.fp_iters_weight = fp_iters_weight
        self.fp_residual_weight = fp_residual_weight
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        
        batch = model_kwargs["batch"]
        labels = batch["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = (loss_counts > 0)
            count_f = valid_metrics.to(torch.float32).sum()

            metrics = {
                "count": count_f,
                "accuracy": torch.where(
                    valid_metrics,
                    (is_correct.to(torch.float32) / loss_divisor).sum(-1),
                    torch.zeros_like(loss_counts, dtype=torch.float32),
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).to(torch.float32).sum(),
            }

        # Losses

        lm_loss = (self.loss_fn(
            outputs["logits"],
            labels,
            ignore_index=IGNORE_LABEL_ID,
            valid_mask=mask,
        ) / loss_divisor).sum()

        metrics["lm_loss"] = lm_loss.detach()

        # Optional fixed point penalties (only active if the model returns these keys)
        fp_iters_loss = torch.zeros((), device=lm_loss.device, dtype=torch.float32)
        fp_residual_loss = torch.zeros((), device=lm_loss.device, dtype=torch.float32)

        if "iters_used" in outputs:
            fp_iters_loss = outputs["iters_used"].to(torch.float32).sum()
            metrics["fp_iters_used_sum"] = fp_iters_loss.detach()

        if "fp_residual_probe" in outputs:
            fp_residual_loss = outputs["fp_residual_probe"].to(torch.float32).sum()
            metrics["fp_residual_probe_sum"] = fp_residual_loss.detach()
        
        if "r_at_stop" in outputs:
            metrics["fp_r_at_stop_sum"] = outputs["r_at_stop"].to(torch.float32).sum().detach()
        
        if "converged" in outputs:
            metrics["fp_converged_count"] = outputs["converged"].to(torch.float32).sum().detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        
        total_loss = lm_loss + (self.fp_residual_weight * fp_residual_loss)
        all_finish = torch.ones((), dtype=torch.bool, device=total_loss.device)

        return new_carry, total_loss, metrics, detached_outputs, all_finish

