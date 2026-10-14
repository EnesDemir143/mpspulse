"""Fixture with no host-device syncs."""

import torch


def train(epochs):
    model = torch.nn.Linear(8, 8).to("mps")
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    total = torch.zeros((), device="mps")
    for epoch in range(epochs):
        x = torch.randn(4, 8, device="mps")
        loss = model(x).sum()
        loss.backward()
        opt.step()
        total = total + loss.detach()
    return total
