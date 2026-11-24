# SPDX-License-Identifier: OpenMDW-1.1
"""Diffusion time conditioning for the WorldFlow generator."""

from math import log

import torch
from torch import nn
from torch.nn import functional as F


class TimestepEmbedder(nn.Module):
    """Project cosine/sine Fourier features into the transformer's hidden space."""

    def __init__(self, hidden_size, frequency_embedding_size=256, bias=True):
        super().__init__()
        self.width = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=bias),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=bias),
        )

    def forward(self, timesteps):
        # Log-spaced angular frequencies run from one toward 1/10000. Match
        # the checkpoint's CPU float32 frequency grid before evaluating phase.
        pairs = self.width // 2
        bands = torch.arange(pairs, dtype=torch.float32, device="cpu")
        omega = (bands * -log(10000.0) / pairs).exp().to(timesteps.device)
        phase = torch.outer(timesteps.float(), omega)
        features = torch.cat((phase.cos(), phase.sin()), dim=1)
        return self.mlp(F.pad(features, (0, self.width % 2)))
