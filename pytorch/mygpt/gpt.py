import torch
import torch.nn as nn

from typing import TypeVar
TensorType = TypeVar('TensorType')


class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.get_keys = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.get_queries = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.get_values = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        keys = self.get_keys(embedded)  # (batch_size, seq_len, attention_dim)
        queries = self.get_queries(embedded)
        values = self.get_values(embedded)

        scores = queries @ torch.transpose(keys, 1, 2)
        B, T, A = keys.shape
        scores = scores / (A ** 0.5)

        pre_mask = torch.tril(torch.ones(T, T))
        mask = pre_mask == 0
        scores = scores.masked_fill(mask, float('-inf'))
        # (batch_size, seq_len, seq_len)
        scores = nn.functional.softmax(scores, dim=2)

        transformed = scores @ values
        # return transformed
        return torch.round(transformed, decimals=4)
