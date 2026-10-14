"""Fixture with deliberate host-device syncs (do not fix)."""

import torch


def train(epochs):
    model = torch.nn.Linear(8, 8).to("mps")
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        x = torch.randn(4, 8, device="mps")
        loss = model(x).sum()
        print(f"epoch {epoch} loss {loss.item()}")
        loss.backward()
        opt.step()
        w = model.weight.cpu()
        arr = model.weight.numpy()
        if bool(loss > 0):
            scale = float(loss)
        idx = torch.tensor([1, 2], device="mps")
    return w, arr, scale, idx
