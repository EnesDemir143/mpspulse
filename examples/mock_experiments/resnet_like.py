"""ResNet-like mock: small CNN train_step on synthetic MPS tensors."""

import torch

torch.manual_seed(0)

_model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 32, 3, padding=1),
    torch.nn.BatchNorm2d(32),
    torch.nn.ReLU(),
    torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
    torch.nn.BatchNorm2d(64),
    torch.nn.ReLU(),
    torch.nn.AdaptiveAvgPool2d(1),
    torch.nn.Flatten(),
    torch.nn.Linear(64, 10),
).to("mps")
_opt = torch.optim.SGD(_model.parameters(), lr=0.01)
_loss_fn = torch.nn.CrossEntropyLoss()
_batch = torch.randn(8, 3, 64, 64, device="mps")
_target = torch.randint(0, 10, (8,), device="mps")


def train_step(batch=None):
    _opt.zero_grad(set_to_none=True)
    loss = _loss_fn(_model(_batch if batch is None else batch), _target)
    loss.backward()
    _opt.step()
    return loss
