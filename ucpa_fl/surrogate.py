from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .datasets import make_loader


class SparseLinearSurrogate(nn.Module):
    """Simple input-space surrogate used for the reproducible xFedAlign-style track."""
    def __init__(self, input_shape, n_classes: int):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.linear = nn.Linear(int(torch.tensor(input_shape).prod().item()), n_classes)

    def forward(self, x):
        return self.linear(x.flatten(1))

    @torch.no_grad()
    def attribution(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(1)
        w = self.linear.weight[targets]
        return (w * flat).abs().reshape_as(x)


def fit_surrogate(task_model, dataset, input_shape, n_classes, cfg, seed: int, device, num_workers: int = 0):
    s = SparseLinearSurrogate(input_shape, n_classes).to(device)
    opt = torch.optim.SGD(s.parameters(), lr=cfg.lr)
    loader = make_loader(dataset, 64, True, seed, num_workers)
    task_model.eval(); s.train()
    T = float(cfg.temperature)
    for _ in range(int(cfg.epochs)):
        for x, _ in loader:
            x = x.to(device)
            with torch.no_grad():
                teacher = F.softmax(task_model(x) / T, dim=1)
            logits = s(x)
            logp = F.log_softmax(logits / T, dim=1)
            kd = F.kl_div(logp, teacher, reduction='batchmean') * (T*T)
            l1 = sum(p.abs().sum() for p in s.parameters())
            loss = kd + float(cfg.l1) * l1
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    s.eval()
    return s
