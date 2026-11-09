"""A compact character-level GPT implementation in PyTorch.

Architecture follows the decoder-only Transformer design. See README.md
for dataset and tutorial attribution.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttentionHead(nn.Module):
    """One causal self-attention head."""
    def __init__(self, embedding_dim: int, head_dim: int, context_length: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_dim, bias=False)
        self.query = nn.Linear(embedding_dim, head_dim, bias=False)
        self.value = nn.Linear(embedding_dim, head_dim, bias=False)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_length, context_length))
        )
        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        keys = self.key(x)
        queries = self.query(x)

        scores = queries @ keys.transpose(-2, -1)
        scores = scores * (keys.size(-1) ** -0.5)
        mask = self.causal_mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)
        values = self.value(x)
        return weights @ values


class MultiHeadCausalAttention(nn.Module):
    """Runs multiple causal attention heads in parallel."""
    def __init__(self, embedding_dim: int, num_heads: int, context_length: int, dropout: float):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        head_dim = embedding_dim // num_heads
        self.heads = nn.ModuleList([
            CausalSelfAttentionHead(embedding_dim, head_dim, context_length, dropout)
            for _ in range(num_heads)
        ])
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.residual_dropout(self.output_projection(combined))


class FeedForwardNetwork(nn.Module):
    """Position-wise MLP used after attention."""
    def __init__(self, embedding_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Pre-layer-normalized Transformer decoder block."""
    def __init__(self, embedding_dim: int, num_heads: int, context_length: int, dropout: float):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadCausalAttention(
            embedding_dim, num_heads, context_length, dropout
        )
        self.feed_forward_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForwardNetwork(embedding_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


class CharacterGPT(nn.Module):
    """Decoder-only GPT that predicts one character at a time."""
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 6,
        context_length: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)
        self.transformer = nn.Sequential(*[
            TransformerBlock(embedding_dim, num_heads, context_length, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.language_model_head = nn.Linear(embedding_dim, vocab_size)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None):
        _, sequence_length = token_ids.shape
        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds context length {self.context_length}"
            )

        positions = torch.arange(sequence_length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.transformer(x)
        logits = self.language_model_head(self.final_norm(x))

        loss = None
        if targets is not None:
            batch_size, seq_len, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.reshape(batch_size * seq_len, vocab_size),
                targets.reshape(batch_size * seq_len),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.context_length:]
            logits, _ = self(context)
            next_token_logits = logits[:, -1, :] / temperature
            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids
