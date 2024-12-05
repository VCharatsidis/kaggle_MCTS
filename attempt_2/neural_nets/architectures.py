import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, DistilBertModel, DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

class ChunkedDistilBertRegression(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, num_chunks):
        """
        Args:
            input_ids: shape (batch_size, num_chunks, seq_length) or (batch_size, seq_length)
            attention_mask: shape (batch_size, num_chunks, seq_length) or (batch_size, seq_length)
            num_chunks: tensor of shape (batch_size) containing number of chunks per sample
        """
        # Check if we're dealing with chunks
        if len(input_ids.shape) == 3:
            batch_size, max_chunks, seq_length = input_ids.shape

            # Reshape to process all chunks at once
            input_ids = input_ids.view(-1, seq_length)
            attention_mask = attention_mask.view(-1, seq_length)

            # Process through DistilBERT
            outputs = self.distilbert(input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token

            # Reshape back to (batch_size, num_chunks, hidden_size)
            pooled_output = pooled_output.view(batch_size, max_chunks, -1)

            # Average across chunks for each sample in the batch
            # Create a mask for valid chunks
            chunk_mask = torch.arange(max_chunks, device=pooled_output.device)[None, :] < num_chunks[:, None]
            chunk_mask = chunk_mask.unsqueeze(-1).float()

            # Masked average
            averaged_output = (pooled_output * chunk_mask).sum(dim=1) / num_chunks.unsqueeze(-1).float()

        else:
            # Single chunk/regular processing
            outputs = self.distilbert(input_ids, attention_mask=attention_mask)
            averaged_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token

        # Pass through regression head
        return self.regressor(averaged_output)