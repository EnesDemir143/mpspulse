"""ViT-like mock: tiny transformer train_step on synthetic MPS tensors."""

import torch

torch.manual_seed(0)

_dim, _heads, _seq = 128, 4, 49
_block = torch.nn.TransformerEncoderLayer(
    d_model=_dim, nhead=_heads, dim_feedforward=256, batch_first=True
).to("mps")
_model = torch.nn.Sequential(torch.nn.Linear(48, _dim), _block, torch.nn.Linear(_dim, 10)).to("mps")
_opt = torch.optim.AdamW(_model.parameters(), lr=1e-3)
_loss_fn = torch.nn.CrossEntropyLoss()
_batch = torch.randn(8, _seq, 48, device="mps")
_target = torch.randint(0, 10, (8, _seq), device="mps")


def train_step(batch=None):
    _opt.zero_grad(set_to_none=True)
    out = _model(_batch if batch is None else batch)
    loss = _loss_fn(out.reshape(-1, 10), _target.reshape(-1))
    loss.backward()
    _opt.step()
    return loss
