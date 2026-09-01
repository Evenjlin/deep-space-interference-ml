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

def train_shift_cnn(model, X_train, y_train, X_val, y_val, device,
                      epochs=30, batch_size=128, lr=1e-3, patience=5, seed=42):
    """Moved from notebook-local code (05_time_shift_cnn.ipynb) for reuse, per modular structure."""
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    Xt, yt = torch.tensor(X_train, device=device), torch.tensor(y_train, device=device)
    Xv, yv = torch.tensor(X_val, device=device), torch.tensor(y_val, device=device)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val, best_state, patience_ct = float("inf"), None, 0
    n = Xt.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        ep_losses = []
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            opt.zero_grad()
            out = model(Xt[idx])
            loss = loss_fn(out, yt[idx])
            loss.backward(); opt.step()
            ep_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_out = model(Xv)
            val_loss = loss_fn(val_out, yv).item()
            val_acc = (val_out.argmax(dim=1) == yv).float().mean().item()

        train_loss = float(np.mean(ep_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_loss < best_val:
            best_val, best_state, patience_ct = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_ct += 1
            if patience_ct >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    model.load_state_dict(best_state)
    return model, history

import torch.nn.functional as F

def decision_aware_loss(output, target, bits_batch, fd_batch, phi_m_batch, k_shift_batch,
                          cfg, n_symbols, mse_weight=1.0, decision_weight=1.0, margin=0.3):
    """
    OUR PROPOSED IMPROVEMENT: augments MSE with a differentiable penalty on
    bit-decision errors, using the SAME de-rotation logic as evaluate.py's
    matched_filter_ber -- directly targets what BER actually depends on
    (sign of the derotated imaginary component), not just aggregate energy.

    Uses real-valued trig (not torch.complex) throughout for autograd safety.
    margin, decision_weight: OUR ASSUMPTION starting hyperparameters, not tuned.
    """
    mse = F.mse_loss(output, target)

    B, _, N = output.shape
    device = output.device

    # Time-shift correction (per-sample k_shift via batched gather)
    idx = (torch.arange(N, device=device).unsqueeze(0) + k_shift_batch.unsqueeze(1)) % N
    idx_exp = idx.unsqueeze(1).expand(-1, 2, -1)
    shifted = torch.gather(output, 2, idx_exp)
    out_real, out_imag = shifted[:, 0, :], shifted[:, 1, :]

    # Frequency/phase de-rotation: multiply by exp(-i*theta), real-valued form
    n = torch.arange(N, device=device, dtype=torch.float32).unsqueeze(0)
    theta = 2 * np.pi * fd_batch.unsqueeze(1) / cfg.Fs * n + phi_m_batch.unsqueeze(1)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    derot_imag = out_imag * cos_t - out_real * sin_t

    # Block-average per symbol, trim first/last (matches evaluate.py convention)
    derot_imag_sym = derot_imag.view(B, n_symbols, cfg.NSPS).mean(dim=2)
    imag_trimmed = derot_imag_sym[:, 1:-1]
    decision_target = 2 * bits_batch[:, 1:-1].float() - 1  # {0,1} -> {-1,+1}

    # Hinge-style margin loss on the decision-critical component
    decision_term = torch.relu(margin - decision_target * imag_trimmed).mean()

    total = mse_weight * mse + decision_weight * decision_term
    return total, mse.item(), decision_term.item()


def train_mitigation_ae_decision_aware(model, X_train, Y_train, bits_train, fd_train, phi_m_train, k_shift_train,
                                          X_val, Y_val, bits_val, fd_val, phi_m_val, k_shift_val,
                                          cfg, n_symbols, device, epochs=100, batch_size=32, lr=3e-5,
                                          patience=15, decision_weight=1.0, margin=0.3, seed=42):
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xt = torch.tensor(X_train, device=device)
    Yt = torch.tensor(Y_train, device=device)
    bits_t = torch.tensor(bits_train, device=device, dtype=torch.long)
    fd_t = torch.tensor(fd_train, device=device, dtype=torch.float32)
    phi_t = torch.tensor(phi_m_train, device=device, dtype=torch.float32)
    k_t = torch.tensor(k_shift_train, device=device, dtype=torch.long)

    Xv = torch.tensor(X_val, device=device)
    Yv = torch.tensor(Y_val, device=device)
    bits_v = torch.tensor(bits_val, device=device, dtype=torch.long)
    fd_v = torch.tensor(fd_val, device=device, dtype=torch.float32)
    phi_v = torch.tensor(phi_m_val, device=device, dtype=torch.float32)
    k_v = torch.tensor(k_shift_val, device=device, dtype=torch.long)

    history = {"train_total": [], "train_mse": [], "train_decision": [], "val_total": []}
    best_val, best_state, patience_ct = float("inf"), None, 0
    n = Xt.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        ep_total, ep_mse, ep_dec = [], [], []
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            opt.zero_grad()
            out = model(Xt[idx])
            loss, mse_val, dec_val = decision_aware_loss(
                out, Yt[idx], bits_t[idx], fd_t[idx], phi_t[idx], k_t[idx],
                cfg, n_symbols, decision_weight=decision_weight, margin=margin)
            loss.backward(); opt.step()
            ep_total.append(loss.item()); ep_mse.append(mse_val); ep_dec.append(dec_val)

        model.eval()
        with torch.no_grad():
            val_out = model(Xv)
            val_loss, _, _ = decision_aware_loss(val_out, Yv, bits_v, fd_v, phi_v, k_v,
                                                    cfg, n_symbols, decision_weight=decision_weight, margin=margin)

        history["train_total"].append(float(np.mean(ep_total)))
        history["train_mse"].append(float(np.mean(ep_mse)))
        history["train_decision"].append(float(np.mean(ep_dec)))
        history["val_total"].append(val_loss.item())

        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}  total={history['train_total'][-1]:.4f}  "
                  f"mse={history['train_mse'][-1]:.4f}  decision={history['train_decision'][-1]:.4f}  "
                  f"val_total={history['val_total'][-1]:.4f}")

        if history["val_total"][-1] < best_val:
            best_val, best_state, patience_ct = history["val_total"][-1], {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_ct += 1
            if patience_ct >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    model.load_state_dict(best_state)
    return model, history