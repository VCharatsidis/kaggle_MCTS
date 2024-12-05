import torch.nn as nn
import torch.nn.functional as F
import torch
import random

class MLP(nn.Module):
    def __init__(self, input_dim, embedding_dim, k=30, temperature=0.1):
        super().__init__()

        # Embedding network
        self.embedding_net = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.Dropout(0.5),
            nn.GELU(),
            nn.BatchNorm1d(embedding_dim),

            nn.Linear(embedding_dim, embedding_dim),
            nn.Dropout(0.5),
            nn.GELU(),
            nn.BatchNorm1d(embedding_dim),

            nn.Linear(embedding_dim, 1),
            nn.Tanh(),

        )

        self.k = k
        self.temperature = temperature

        # Support set storage
        self.support_embeddings = None
        self.support_labels = None

    def forward(self, x):
        return self.embedding_net(x)