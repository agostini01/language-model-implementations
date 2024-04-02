import torch
import torch.nn as nn

from torchtyping import TensorType


class GPT(nn.Module):

    def __init__(self, vocab_size: int, context_len: int, embedding_dim: int, attention_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(context_len, embedding_dim)
        self.blocks = nn.Sequential()
        for _ in range(num_blocks):
            self.blocks.append(self.TransformerBlock(
                embedding_dim, attention_dim, num_heads))
        self.final_ln = nn.LayerNorm(embedding_dim)
        self.final_projection = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context: TensorType[int]) -> TensorType[float]:
        token_embedded = self.token_embedding(context)
        B, T, E = token_embedded.shape
        position_embedded = self.pos_embedding(torch.arange(T))
        total_embedded = token_embedded + position_embedded

        not_normalized = self.final_projection(self.final_ln(
            self.blocks(total_embedded)))
        probs = nn.functional.softmax(not_normalized, dim=-1)
        return probs

    class TransformerBlock(nn.Module):

        def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
            super().__init__()

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

                self.heads = nn.ModuleList()
                for _ in range(num_heads):
                    self.heads.append(self.SingleHeadAttention(
                        embedding_dim, attention_dim//num_heads))

            def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                # B, T, Head size >>> B, T, Attention dim
                outputs = [head(embedded) for head in self.heads]
                cated = torch.cat(outputs, dim=-1)
                return cated

            class SingleHeadAttention(nn.Module):

                def __init__(self, embedding_dim: int, attention_dim: int):
                    super().__init__()

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
                    return transformed

        class MLPNet(nn.Module):

            def __init__(self, embedding_dim: int):
                super().__init__()

                self.up_projection = nn.Linear(
                    embedding_dim, embedding_dim * 4)
                self.relu = nn.ReLU()
                self.down_projection = nn.Linear(
                    embedding_dim * 4, embedding_dim)
                self.dropout = nn.Dropout(0.2)  # using p = 0.2

            def forward(self, x: TensorType[float]) -> TensorType[float]:

                return self.dropout(self.down_projection(self.relu(self.up_projection(x))))
