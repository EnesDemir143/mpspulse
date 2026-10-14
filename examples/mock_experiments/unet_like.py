"""UNet-like mock: tiny encoder-decoder train_step on synthetic MPS tensors."""

import torch

torch.manual_seed(0)

_model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 16, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(16, 32, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.ConvTranspose2d(32, 16, 2, stride=2),
    torch.nn.ReLU(),
    torch.nn.Conv2d(16, 2, 1),
).to("mps")
_opt = torch.optim.SGD(_model.parameters(), lr=0.01)
_loss_fn = torch.nn.CrossEntropyLoss()
_batch = torch.randn(4, 3, 64, 64, device="mps")
_target = torch.randint(0, 2, (4, 64, 64), device="mps")


def train_step(batch=None):
    _opt.zero_grad(set_to_none=True)
    loss = _loss_fn(_model(_batch if batch is None else batch), _target)
    loss.backward()
    _opt.step()
    return loss
