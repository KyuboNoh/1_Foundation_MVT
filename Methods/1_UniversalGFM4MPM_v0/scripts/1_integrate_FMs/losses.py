from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


class SupConLoss(torch.nn.Module):
    """Supervised contrastive loss focusing on positive labels."""

    def __init__(self, temperature: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        mask = labels.unsqueeze(0) * labels.unsqueeze(1)
        positives = mask.bool()
        logits = torch.matmul(embeddings, embeddings.T) / self.temperature
        logits = logits - torch.max(logits, dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * (~torch.eye(labels.size(0), device=device).bool())
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + self.eps)
        loss = 0.0
        num_pos = positives.sum(dim=1)
        valid = num_pos > 0
        if valid.any():
            mean_log_prob_pos = (log_prob * positives).sum(dim=1) / (num_pos + self.eps)
            loss = -(mean_log_prob_pos[valid]).mean()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        return loss


class PrototypeLoss(torch.nn.Module):
    """Encourages embeddings to stay close to their dataset-specific prototypes."""

    def __init__(self):
        super().__init__()

    def forward(self, embeddings: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
        return ((embeddings - prototype) ** 2).sum(dim=1).mean()


class NNPU(torch.nn.Module):
    """Non-negative PU loss following Kiryo et al. (2017)."""

    def __init__(self, class_prior: float, beta: float = 0.0, gamma: float = 1.0):
        super().__init__()
        self.class_prior = class_prior
        self.beta = beta
        self.gamma = gamma

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        preds = preds.view(-1)
        labels = labels.view(-1)
        positives = preds[labels == 1]
        unlabeled = preds[labels == 0]
        if positives.numel() == 0 or unlabeled.numel() == 0:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        loss_pos = F.binary_cross_entropy_with_logits(positives, torch.ones_like(positives))
        loss_neg = F.binary_cross_entropy_with_logits(unlabeled, torch.zeros_like(unlabeled))
        u_neg = (unlabeled.sigmoid()).mean()
        pn_loss = -self.class_prior * F.logsigmoid(-positives).mean()
        nn_loss = F.logsigmoid(-unlabeled).mean()
        risk = self.class_prior * loss_pos + torch.clamp(nn_loss - self.class_prior * (1 - self.class_prior) * u_neg, min=-self.beta)
        return self.gamma * risk + (1 - self.gamma) * pn_loss


class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        logits = anchor @ positive.T / self.temperature
        labels = torch.arange(anchor.shape[0], device=anchor.device)
        return F.cross_entropy(logits, labels)


class SymmetricKLLoss(torch.nn.Module):
    def __init__(self, reduction: str = "batchmean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logit_a: torch.Tensor, logit_b: torch.Tensor) -> torch.Tensor:
        probs_a = F.log_softmax(logit_a, dim=-1)
        probs_b = F.log_softmax(logit_b, dim=-1)
        p_a = probs_a.exp()
        p_b = probs_b.exp()
        kl_ab = F.kl_div(probs_a, p_b, reduction=self.reduction)
        kl_ba = F.kl_div(probs_b, p_a, reduction=self.reduction)
        return 0.5 * (kl_ab + kl_ba)


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversal(torch.nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class MMDLoss(torch.nn.Module):
    def __init__(self, kernel_multiplier: float = 2.0, kernel_count: int = 5):
        super().__init__()
        self.kernel_multiplier = kernel_multiplier
        self.kernel_count = kernel_count

    def _gaussian_kernel(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = torch.cat([source, target], dim=0)
        total_square = (total ** 2).sum(dim=1, keepdim=True)
        exponent = total_square + total_square.T - 2 * (total @ total.T)
        kernel_bandwidth = torch.sum(exponent.data) / (total.shape[0] ** 2 - total.shape[0])
        kernel_bandwidth /= self.kernel_multiplier ** (self.kernel_count // 2)
        bandwidth_list = [kernel_bandwidth * (self.kernel_multiplier ** i) for i in range(self.kernel_count)]
        kernel_val = sum(torch.exp(-exponent / bandwidth) for bandwidth in bandwidth_list)
        return kernel_val

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = source.size(0)
        kernels = self._gaussian_kernel(source, target)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        return torch.mean(XX + YY - XY - YX)
