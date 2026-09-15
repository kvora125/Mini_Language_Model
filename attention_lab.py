"""
Attention Lab - a mini language-model pipeline built with NumPy only.

    text representation -> model training -> self-attention -> training diagnostics

Every calculation (tokenisation, embedding lookup, forward pass, cross-entropy,
back-propagation, scaled dot-product attention, checkpoint selection) is written
out by hand so it can be inspected line by line.

Run:  python attention_lab.py
Outputs: console report, loss_curve.png, attention_heatmap.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # render to file; no display needed
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=120)

# ---------------------------------------------------------------------------
# Data (all data used by the lab lives here)
# ---------------------------------------------------------------------------
TASK1_TEXT = "the cat sat on a mat"                          # 6 unique tokens
VALIDATION_TEXT = "a cat sat on a mat"                       # held-out text, same vocabulary
ATTENTION_SENTENCE = "i went to the bank to deposit my money"  # 9 tokens

EMBED_DIM = 8
LEARNING_RATES = (0.1, 1.0)
EPOCHS = 200
DIAGNOSTIC_LR = 0.1
DIAGNOSTIC_EPOCHS = 300
OVERFIT_PATIENCE = 5


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def softmax(x, axis=-1):
    """Numerically stable softmax: subtract the row max before exponentiating."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


# --- Task 1: Text Representation ---
def tokenize(text):
    """Lower-case whitespace tokenizer."""
    return text.lower().split()


def build_vocab(tokens):
    """Map each unique token to an integer id, in order of first appearance."""
    stoi = {}
    for tok in tokens:
        if tok not in stoi:
            stoi[tok] = len(stoi)
    itos = {i: tok for tok, i in stoi.items()}
    return stoi, itos


def encode(tokens, stoi):
    return np.array([stoi[t] for t in tokens], dtype=int)


def create_embeddings(vocab_size, dim, seed=SEED, scale=0.1):
    """Random embedding matrix: one row (vector) per token id."""
    rng = np.random.RandomState(seed)
    return rng.randn(vocab_size, dim) * scale


banner("Task 1: Text Representation")
task1_tokens = tokenize(TASK1_TEXT)
stoi, itos = build_vocab(task1_tokens)
task1_ids = encode(task1_tokens, stoi)
vocab_size = len(stoi)
embedding_matrix = create_embeddings(vocab_size, EMBED_DIM)

assert vocab_size == 6, "Task 1 vocabulary must contain 6 tokens"
assert embedding_matrix.shape == (6, 8), "Embedding matrix must be (6, 8)"

print(f"Text           : {TASK1_TEXT!r}")
print(f"Tokens         : {task1_tokens}")
print(f"Vocabulary     : {stoi}")
print(f"Vocab size     : {vocab_size}")
print(f"Token ids      : {task1_ids}")
print(f"Embedding shape: {embedding_matrix.shape}")
print("Embedding matrix (row i = vector for token id i):")
for i in range(vocab_size):
    print(f"  {itos[i]:>4} -> {embedding_matrix[i]}")
sequence_vectors = embedding_matrix[task1_ids]  # embedding lookup = row indexing
print(f"Embedded sequence shape: {sequence_vectors.shape} (tokens x embedding dim)")


# --- Task 2: Bigram Training ---
# Model: next-token logits = E[current_token] @ W + b
#   E : (V, D) token embeddings (initialised from Task 1)
#   W : (D, V) output projection,  b : (V,) output bias
def init_bigram_params(embeddings, seed=SEED):
    rng = np.random.RandomState(seed)
    vocab, dim = embeddings.shape
    return {
        "E": embeddings.copy(),
        "W": rng.randn(dim, vocab) * 0.1,
        "b": np.zeros(vocab),
    }


def make_bigrams(ids):
    """Input = token at position t, target = token at position t + 1."""
    return ids[:-1], ids[1:]


def bigram_forward(params, x, y):
    h = params["E"][x]                       # (N, D) embedding lookup
    logits = h @ params["W"] + params["b"]   # (N, V)
    probs = softmax(logits)                  # (N, V)
    loss = -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12))  # cross-entropy
    return loss, (h, probs)


def bigram_backward(params, x, y, cache):
    h, probs = cache
    n = len(y)
    dlogits = probs.copy()
    dlogits[np.arange(n), y] -= 1.0          # d(CE)/d(logits) = softmax - one_hot
    dlogits /= n
    grads = {
        "W": h.T @ dlogits,
        "b": dlogits.sum(axis=0),
        "E": np.zeros_like(params["E"]),
    }
    dh = dlogits @ params["W"].T
    np.add.at(grads["E"], x, dh)             # scatter gradients back to embedding rows
    return grads


def train_bigram(params, x, y, lr, epochs, x_val=None, y_val=None):
    """Full-batch gradient descent. Returns train losses, val losses, param snapshots."""
    train_losses, val_losses, snapshots = [], [], []
    for _ in range(epochs):
        loss, cache = bigram_forward(params, x, y)
        grads = bigram_backward(params, x, y, cache)
        for name in params:
            params[name] -= lr * grads[name]
        # record the loss of the *updated* weights so epoch k == weights after k updates
        train_losses.append(bigram_forward(params, x, y)[0])
        if x_val is not None:
            val_losses.append(bigram_forward(params, x_val, y_val)[0])
        snapshots.append({k: v.copy() for k, v in params.items()})
    return train_losses, val_losses, snapshots


banner("Task 2: Bigram Training")
x_train, y_train = make_bigrams(task1_ids)
print("Training pairs (input -> target):")
for a, b in zip(x_train, y_train):
    print(f"  {itos[a]:>4} -> {itos[b]}")

lr_results = {}
for lr in LEARNING_RATES:
    params = init_bigram_params(embedding_matrix)   # identical start for each run
    initial_loss = bigram_forward(params, x_train, y_train)[0]
    losses, _, _ = train_bigram(params, x_train, y_train, lr, EPOCHS)
    lr_results[lr] = {"initial": initial_loss, "losses": losses, "params": params}
    assert losses[-1] < initial_loss, f"Loss did not decrease for lr={lr}"
    print(f"\nlr = {lr}")
    print(f"  initial loss : {initial_loss:.4f}  (uniform guess = ln(6) = {np.log(6):.4f})")
    for epoch in (1, 10, 50, 100, EPOCHS):
        print(f"  epoch {epoch:>4}   : {losses[epoch - 1]:.4f}")
    print(f"  reduction    : {100 * (1 - losses[-1] / initial_loss):.1f}%")

best_lr = min(LEARNING_RATES, key=lambda lr: lr_results[lr]["losses"][-1])
print(f"\nLower final loss with lr = {best_lr}. Next-token predictions:")
_, (_, probs) = bigram_forward(lr_results[best_lr]["params"], x_train, y_train)
for a, row in zip(x_train, probs):
    pred = int(np.argmax(row))
    print(f"  after {itos[a]!r:>6} -> {itos[pred]!r:<6} (p = {row[pred]:.3f})")


# --- Task 3: Self-Attention ---
def positional_encoding(seq_len, dim):
    """Sinusoidal positions (Vaswani et al., 2017) so repeated words differ by position."""
    pos = np.arange(seq_len)[:, None]
    i = np.arange(dim)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / dim)
    return np.where(i % 2 == 0, np.sin(angle), np.cos(angle))


def self_attention(X, W_q, W_k, W_v):
    """Scaled dot-product attention: softmax(Q K^T / sqrt(d_k)) V."""
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)          # (T, T) similarity of every query to every key
    weights = softmax(scores, axis=-1)       # each row sums to 1
    return weights @ V, weights, scores


banner("Task 3: Self-Attention")
attn_tokens = tokenize(ATTENTION_SENTENCE)
attn_stoi, attn_itos = build_vocab(attn_tokens)       # Task 3's own vocabulary
attn_ids = encode(attn_tokens, attn_stoi)
# own embeddings; unit scale so token identity is not drowned out by the positional encoding
attn_embeddings = create_embeddings(len(attn_stoi), EMBED_DIM, seed=SEED, scale=1.0)
seq_len = len(attn_tokens)

rng = np.random.RandomState(SEED)
W_q = rng.randn(EMBED_DIM, EMBED_DIM) / np.sqrt(EMBED_DIM)
W_k = rng.randn(EMBED_DIM, EMBED_DIM) / np.sqrt(EMBED_DIM)
W_v = rng.randn(EMBED_DIM, EMBED_DIM) / np.sqrt(EMBED_DIM)

X = attn_embeddings[attn_ids] + positional_encoding(seq_len, EMBED_DIM)
attn_output, attn_weights, attn_scores = self_attention(X, W_q, W_k, W_v)

assert attn_weights.shape == (9, 9), "Attention matrix must be 9 x 9"
assert np.allclose(attn_weights.sum(axis=1), 1.0), "Attention rows must sum to 1"

print(f"Sentence        : {ATTENTION_SENTENCE!r}")
print(f"Tokens          : {attn_tokens}")
print(f"Vocabulary      : {attn_stoi}  (size {len(attn_stoi)})")
print(f"Token ids       : {attn_ids}")
print(f"Embedding shape : {attn_embeddings.shape}")
print(f"Input X shape   : {X.shape}  (token embedding + positional encoding)")
print(f"Attention shape : {attn_weights.shape}")
print(f"Output shape    : {attn_output.shape}")

col_w = 8
print("\nAttention matrix (row = query token, column = key token):")
print(" " * 9 + "".join(f"{t:>{col_w}}" for t in attn_tokens))
for tok, row in zip(attn_tokens, attn_weights):
    print(f"{tok:>9}" + "".join(f"{w:>{col_w}.3f}" for w in row))

bank_pos = attn_tokens.index("bank")
bank_row = attn_weights[bank_pos]
print(f"\nAttention weights for 'bank' (position {bank_pos}), sum = {bank_row.sum():.3f}:")
for j in np.argsort(-bank_row):
    bar = "#" * int(round(bank_row[j] * 100))
    print(f"  bank -> {attn_tokens[j]:<8} (pos {j}): {bank_row[j]:.4f} {bar}")
print("Note: W_q, W_k, W_v are random (untrained), so these weights show the mechanics,")
print("not learned meaning. Training would push 'bank' toward context words like 'deposit'.")

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(attn_weights, cmap="viridis")
ax.set_xticks(range(seq_len))
ax.set_yticks(range(seq_len))
ax.set_xticklabels(attn_tokens, rotation=45, ha="right")
ax.set_yticklabels(attn_tokens)
ax.set_xlabel("Key (attended to)")
ax.set_ylabel("Query (attending)")
ax.set_title("Self-attention weights")
for i in range(seq_len):
    for j in range(seq_len):
        ax.text(j, i, f"{attn_weights[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if attn_weights[i, j] < attn_weights.max() * 0.6 else "black")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig("attention_heatmap.png", dpi=150)
plt.close(fig)
print("\nSaved attention_heatmap.png")


# --- Task 4: Training Diagnostics ---
def find_best_checkpoint(val_losses):
    """Best checkpoint = epoch with the lowest validation loss (1-indexed)."""
    idx = int(np.argmin(val_losses))
    return idx + 1, val_losses[idx]


def find_overfitting_point(train_losses, val_losses, patience=OVERFIT_PATIENCE):
    """
    First epoch after which validation loss rises for `patience` consecutive epochs
    while training loss keeps falling - the model is memorising, not generalising.
    Returns the 1-indexed epoch where the divergence starts, or None.
    """
    for t in range(1, len(val_losses) - patience + 1):
        val_up = all(val_losses[k] > val_losses[k - 1] for k in range(t, t + patience))
        train_down = all(train_losses[k] < train_losses[k - 1] for k in range(t, t + patience))
        if val_up and train_down:
            return t + 1
    return None


banner("Task 4: Training Diagnostics")
val_tokens = tokenize(VALIDATION_TEXT)
assert all(t in stoi for t in val_tokens), "Validation text must use the Task 1 vocabulary"
x_val, y_val = make_bigrams(encode(val_tokens, stoi))

print(f"Train text      : {TASK1_TEXT!r}")
print(f"Validation text : {VALIDATION_TEXT!r}")
print(f"Learning rate   : {DIAGNOSTIC_LR}   Epochs: {DIAGNOSTIC_EPOCHS}")

diag_params = init_bigram_params(embedding_matrix)
train_losses, val_losses, snapshots = train_bigram(
    diag_params, x_train, y_train, DIAGNOSTIC_LR, DIAGNOSTIC_EPOCHS, x_val, y_val)

best_epoch, best_val = find_best_checkpoint(val_losses)
best_checkpoint = snapshots[best_epoch - 1]
overfit_epoch = find_overfitting_point(train_losses, val_losses)

print("\nEpoch   train_loss   val_loss")
for epoch in sorted({1, 5, 10, 20, 50, 100, 200, DIAGNOSTIC_EPOCHS, best_epoch}):
    marker = "  <- best checkpoint" if epoch == best_epoch else ""
    print(f"{epoch:>5}   {train_losses[epoch - 1]:>10.4f}   {val_losses[epoch - 1]:>8.4f}{marker}")

print(f"\nBest checkpoint      : epoch {best_epoch} (val loss {best_val:.4f}, "
      f"train loss {train_losses[best_epoch - 1]:.4f})")
if overfit_epoch is not None:
    print(f"Possible overfitting : starts at epoch {overfit_epoch} - val loss rises for "
          f"{OVERFIT_PATIENCE}+ epochs while train loss keeps falling")
else:
    print("Possible overfitting : not detected within the training run")
print(f"Final epoch          : train {train_losses[-1]:.4f}, val {val_losses[-1]:.4f}, "
      f"gap {val_losses[-1] - train_losses[-1]:.4f}")
print(f"Early stopping at epoch {best_epoch} would save {DIAGNOSTIC_EPOCHS - best_epoch} epochs "
      f"and keep val loss {val_losses[-1] - best_val:.4f} lower than the final weights.")

# sanity check: reloading the best checkpoint reproduces its validation loss
reloaded_val = bigram_forward(best_checkpoint, x_val, y_val)[0]
assert np.isclose(reloaded_val, best_val)

# ---------------------------------------------------------------------------
# loss_curve.png: Task 2 learning-rate comparison + Task 4 diagnostics
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

for lr in LEARNING_RATES:
    curve = [lr_results[lr]["initial"]] + lr_results[lr]["losses"]
    ax1.plot(range(len(curve)), curve, label=f"lr = {lr}")
ax1.set_title("Task 2: Bigram training loss by learning rate")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-entropy loss")
ax1.legend()
ax1.grid(alpha=0.3)

epochs_axis = np.arange(1, DIAGNOSTIC_EPOCHS + 1)
ax2.plot(epochs_axis, train_losses, label="Train loss")
ax2.plot(epochs_axis, val_losses, label="Validation loss")
ax2.axvline(best_epoch, color="green", linestyle="--",
            label=f"Best checkpoint (epoch {best_epoch})")
ax2.scatter([best_epoch], [best_val], color="green", zorder=3)
if overfit_epoch is not None:
    ax2.axvspan(overfit_epoch, DIAGNOSTIC_EPOCHS, color="red", alpha=0.08,
                label=f"Overfitting (from epoch {overfit_epoch})")
ax2.set_title(f"Task 4: Train vs validation loss (lr = {DIAGNOSTIC_LR})")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Cross-entropy loss")
ax2.legend()
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("loss_curve.png", dpi=150)
plt.close(fig)
print("\nSaved loss_curve.png")

banner("Attention Lab complete")
