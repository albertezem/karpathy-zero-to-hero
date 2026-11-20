import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import os
from model import GPTConfig, GPTLanguageModel
from tokenizer import CharTokenizer

# hyperparameters
batch_size = 64     # how many independent sequences we process in parallel, B
block_size = 256       # maximum context length for predictions, T
n_embd = 384 # embedding vector size, C
n_head = 6
n_layer = 6
dropout = 0.0
max_iters = 0
eval_interval = 1000
learning_rate = 3e-4
eval_iters = 100
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
resume = True # continue from ckpt_path if it exists; max_iters is then the number of extra steps
ckpt_path = 'nanogpt/output/ckpt.pt'
# ------------

torch.manual_seed(1337)

def load_ascii(path):
    raw = np.fromfile(path, dtype=np.uint8)
    return raw[raw < 128] # drop non-ASCII bytes (rare in TinyStories)

train_bytes = load_ascii('nanogpt/input/TinyStoriesV2-GPT4-train.txt')
val_bytes = load_ascii('nanogpt/input/TinyStoriesV2-GPT4-valid.txt')

byte_vals = np.unique(train_bytes) # sorted, so chars stays in byte order
vocab_size = len(byte_vals)
chars = [chr(b) for b in byte_vals.tolist()]
tok = CharTokenizer(chars)

# bulk-encode the dataset: byte value -> token id, the same mapping as tok.encode
# but vectorized; numpy can index with uint8 directly, so no big int64 copy
lut = np.zeros(128, dtype=np.uint8)
lut[byte_vals] = np.arange(vocab_size, dtype=np.uint8)
train_data = torch.from_numpy(lut[train_bytes])
val_data = torch.from_numpy(lut[val_bytes])
del train_bytes, val_bytes


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).long()
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).long()
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    gpt.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = gpt(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    gpt.train()
    return out

xb, yb = get_batch('train')
print(xb.shape, yb.shape)
print(xb.device, yb.device)

config = GPTConfig(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd,
                   n_head=n_head, n_layer=n_layer, dropout=dropout)
gpt = GPTLanguageModel(config)
gpt.to(device)
print(sum(p.numel() for p in gpt.parameters()))
optimizer = torch.optim.AdamW(gpt.parameters(), lr=learning_rate)

def save_model(model, step, best_val):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'best_val': best_val,
        'config': {
            'vocab_size': vocab_size, 'block_size': block_size, 'n_embd': n_embd,
            'n_head': n_head, 'n_layer': n_layer, 'dropout': dropout,
            },
        'chars': chars,
        }
    torch.save(checkpoint, ckpt_path)
    print(f"saved checkpoint at step {step} (val {best_val:.4f})")

start_step = 0
best_val = float('inf')
if resume and os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    assert ckpt['chars'] == chars, "checkpoint was trained with a different vocab"
    gpt.load_state_dict(ckpt['model'])
    if 'optimizer' in ckpt:  # older checkpoints only have weights
        optimizer.load_state_dict(ckpt['optimizer'])
    start_step = ckpt.get('step', -1) + 1
    best_val = ckpt.get('best_val', float('inf'))
    print(f"resumed from step {start_step}, best val {best_val}")
torch.manual_seed(1337 + start_step)

gpt.train()

for steps in range(start_step, start_step + max_iters):
  xb, yb = get_batch('train')

  logits, loss = gpt(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

  if (steps % eval_interval == 0) or (steps == start_step + max_iters - 1):
    losses = estimate_loss()
    print(steps, losses['train'], losses['val'])

    if losses['val'] < best_val:
        best_val = losses['val']
        save_model(gpt, steps, best_val)

