import torch
import torch.nn as nn

from torchtyping import TensorType


class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)

        self.mha = self.MultiHeadAttention(embedding_dim,
                                           attention_dim, num_heads)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ffn = self.MLPNet(embedding_dim)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        first_part = x + self.mha(self.ln1(x))
        second_part = first_part + self.ffn(self.ln2(first_part))
        # return second_part
        return torch.round(second_part, decimals=4)

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
                self.get_keys = nn.Linear(
                    embedding_dim, attention_dim, bias=False)
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

    class MLPNet(nn.Module):

        def __init__(self, embedding_dim: int):
            super().__init__()
            torch.manual_seed(0)
            self.up_projection = nn.Linear(embedding_dim, embedding_dim * 4)
            self.relu = nn.ReLU()
            self.down_projection = nn.Linear(embedding_dim * 4, embedding_dim)
            self.dropout = nn.Dropout(0.2)  # using p = 0.2

        def forward(self, x: TensorType[float]) -> TensorType[float]:
            torch.manual_seed(0)
            return self.dropout(self.down_projection(self.relu(self.up_projection(x))))
