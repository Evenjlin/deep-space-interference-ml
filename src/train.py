"""
Training loop for the MLP autoencoder detector, with debug-ladder helpers
per the master prompt's §15 (small subset -> one batch -> one epoch -> full run).
"""
import numpy as np
import torch
import torch.nn as nn
from src.model import vectorize


def vectors_from_samples(samples):
    return np.stack([vectorize(x) for x in samples]).astype(np.float32)


def debug_ladder(model, train_vecs, device):
    """TEST 1-4 from master prompt: tiny subset, one batch, one epoch, loss sanity."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # TEST 1: tiny subset
    tiny = torch.tensor(train_vecs[:8], device=device)
    out = model(tiny)
    assert out.shape == tiny.shape, f"Shape mismatch: {out.shape} vs {tiny.shape}"
    print("TEST 1 (tiny subset forward pass): OK, shape", out.shape)

    # TEST 2: one batch, one gradient step
    loss = loss_fn(out, tiny)
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f"TEST 2 (one batch, one grad step): OK, loss={loss.item():.4f}")

    # TEST 3+4: one epoch on a small slice, confirm loss decreases
    small = torch.tensor(train_vecs[:200], device=device)
    losses = []
    for _ in range(5):
        opt.zero_grad()
        out = model(small)
        loss = loss_fn(out, small)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print("TEST 3/4 (5 steps on small slice, loss should trend down):", 
          [f"{l:.4f}" for l in losses])
    assert losses[-1] < losses[0], "Loss did not decrease -- something is wrong before a full run"
    print("Debug ladder PASSED. Safe to proceed to full training.")


def train_autoencoder(model, train_vecs, val_vecs, device, epochs=50, 
                       batch_size=64, lr=1e-3, patience=5, seed = 42):
    """Full training loop with early stopping and loss history logging."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_t = torch.tensor(train_vecs, device=device)
    val_t = torch.tensor(val_vecs, device=device)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    patience_counter = 0

    n = train_t.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_losses = []
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            batch = train_t[idx]
            opt.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch)
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_t), val_t).item()

        train_loss = float(np.mean(epoch_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch+1}/{epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    model.load_state_dict(best_state)
    return model, history