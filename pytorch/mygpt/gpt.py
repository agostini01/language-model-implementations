import torch
import torch.nn as nn

from typing import TypeVar
TensorType = TypeVar('TensorType')


class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)

        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(self.SingleHeadAttention(
                embedding_dim, attention_dim//num_heads))

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # B, T, Head size >>> B, T, Attention dim
        outputs = [head(embedded) for head in self.heads]
        cated = torch.cat(outputs, dim=-1)
        # return cated
        return torch.round(cated, decimals=4)

    class SingleHeadAttention(nn.Module):

        def __init__(self, embedding_dim: int, attention_dim: int):
            super().__init__()
            torch.manual_seed(0)
            self.get_keys = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.get_queries = nn.Linear(
                embedding_dim, attention_dim, bias=False)
            self.get_values = nn.Linear(
                embedding_dim, attention_dim, bias=False)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            # (batch_size, seq_len, attention_dim)
            keys = self.get_keys(embedded)
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
