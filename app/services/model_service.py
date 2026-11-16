"""The tiny GPT: a nanoGPT-style character-level transformer.

Architecture: token embedding + learned positional embedding, N decoder
blocks (causal self-attention + GELU MLP), final LayerNorm, and a head
tied to the token embedding. No dropout by default — the corpus is
small and the model is tiny, so regularisation comes from size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """One transformer block: attention, then MLP, each with a residual."""

    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with a causal attention mask."""
        b, t, c = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(b, t, c)
        x = x + self.drop(self.attn_out(attn))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class TinyGPT(nn.Module):
    """Character-level GPT with weight tying between embedding and head."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layer: int = 4,
        n_head: int = 4,
        context_len: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.context_len = context_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Small random init for linear/embedding weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """Return logits, and the cross-entropy loss when targets given."""
        _, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate_step(self, idx: torch.Tensor) -> torch.Tensor:
        """One forward pass for generation; returns next-token logits."""
        idx = idx[:, -self.context_len :]
        logits, _ = self.forward(idx)
        return logits[:, -1, :]


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count (weight tying means the head is free)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_config_dict(
    vocab_size: int,
    d_model: int,
    n_layer: int,
    n_head: int,
    context_len: int,
    dropout: float,
) -> dict:
    """The config block stored in checkpoints and run records."""
    return {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_layer": n_layer,
        "n_head": n_head,
        "context_len": context_len,
        "dropout": dropout,
    }


def build_model(config: dict) -> TinyGPT:
    """Build a TinyGPT from a config dict (as stored in checkpoints)."""
    return TinyGPT(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        context_len=config["context_len"],
        dropout=config.get("dropout", 0.0),
    )
