import torch

from model import (
    SiameseNetwork,
    cosine_similarity
)

model = SiameseNetwork()

x1 = torch.randn(
    8,
    3,
    224,
    224
)

x2 = torch.randn(
    8,
    3,
    224,
    224
)

emb1, emb2 = model(
    x1,
    x2
)

print(emb1.shape)

print(emb2.shape)

sim = cosine_similarity(
    emb1,
    emb2
)

print(sim.shape)