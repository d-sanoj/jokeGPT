"""Model: shapes, parameter count, loss is finite and learnable."""

import torch

from app.services.model_service import TinyGPT, count_parameters


def test_forward_shapes():
    """Logits have shape (batch, time, vocab)."""
    model = TinyGPT(
        vocab_size=95, d_model=64, n_layer=2, n_head=2, context_len=32
    )
    idx = torch.randint(0, 95, (4, 32))
    logits, loss = model(idx, idx)
    assert logits.shape == (4, 32, 95)
    assert loss.item() > 0


def test_parameter_count():
    """The main config is ~0.8M parameters."""
    model = TinyGPT(
        vocab_size=95, d_model=128, n_layer=4, n_head=4, context_len=128
    )
    params = count_parameters(model)
    assert 700_000 < params < 1_000_000


def test_loss_decreases_on_tiny_overfit():
    """A few steps on one batch must reduce the loss (sanity)."""
    torch.manual_seed(0)
    model = TinyGPT(
        vocab_size=20, d_model=32, n_layer=1, n_head=2, context_len=16
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    idx = torch.randint(0, 20, (8, 16))
    _, first = model(idx, idx)
    for _ in range(30):
        _, loss = model(idx, idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    _, last = model(idx, idx)
    assert last.item() < first.item()
