import torch.nn as nn
import torch.nn.functional as F
import torch
import random

class NeuralKNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, k, temperature):
        super().__init__()

        # Embedding network
        self.embedding_net = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.GELU(),

            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),

            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid(),

        )

        self.k = k
        self.temperature = temperature

        # Support set storage
        self.support_embeddings = None
        self.support_labels = None

    def compute_embeddings(self, x):
        return self.embedding_net(x)

    def compute_distances(self, query_embeddings, support_embeddings):
        # Compute pairwise euclidean distances
        # Shape: [query_size, support_size]
        return torch.cdist(query_embeddings, support_embeddings, p=2)

    def forward(self, x):
        if self.support_embeddings is None:
            raise RuntimeError("Must call set_support before inference!")

        # Get embeddings for query points
        query_embeddings = self.compute_embeddings(x)

        # Compute distances to support set
        distances = self.compute_distances(query_embeddings, self.support_embeddings)

        mask = torch.isclose(
            query_embeddings.unsqueeze(1),  # [query_size, 1, embedding_dim]
            self.support_embeddings.unsqueeze(0),  # [1, support_size, embedding_dim]
            rtol=1e-5,
            atol=1e-5
        ).all(dim=-1)  # [query_size, support_size]

        # print(mask)
        # input()

        # Set masked distances to infinity
        distances = distances.masked_fill(mask, float('inf'))

        # Get k nearest neighbors
        distances, indices = torch.topk(distances, k=self.k, largest=False, dim=1)

        # Get labels of nearest neighbors
        neighbor_labels = self.support_labels[indices]
        neighbor_labels = torch.squeeze(neighbor_labels, dim=-1)

        # Convert distances to weights using softmax
        weights = F.softmax(-distances / self.temperature, dim=1)

        if random.randint(1, 5000) == 1:
            print(weights[0].max())
            print(weights[0].min())
            print(torch.topk(weights[0], k=10))

        # Weighted sum of neighbor labels
        predictions = torch.sum(weights * neighbor_labels, dim=1)

        return predictions

    def set_support(self, x, y, batch_size=256):
        """Set the support set for inference"""
        # with torch.no_grad():
        #     self.support_embeddings = self.compute_embeddings(x)
        #     self.support_labels = y

        n_samples = len(x)
        embeddings_list = []

        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_x = x[start_idx:end_idx]
                batch_embeddings = self.compute_embeddings(batch_x)
                embeddings_list.append(batch_embeddings)

        # Concatenate all batched embeddings
        self.support_embeddings = torch.cat(embeddings_list, dim=0)
        self.support_labels = y
