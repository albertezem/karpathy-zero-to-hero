from typing import Any

import torch
from model import GPTConfig, GPTLanguageModel, LoRALinear, apply_lora
from tokenizer import CharTokenizer
import torch.nn as nn

# fine-tuning settings (the architecture comes from the checkpoint's config)
batch_size = 64
eval_iters = 100
r = 8
alpha = 16
max_iters = 3000
eval_interval = 100
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
ckpt_path = 'nanogpt/output/ckpt.pt'
# ------------

torch.manual_seed(1337)

ckpt = torch.load(ckpt_path, map_location=device)
config = GPTConfig(**ckpt['config'])
gpt = GPTLanguageModel(config)
gpt.load_state_dict(ckpt['model'])
gpt.to(device)
print(sum(p.numel() for p in gpt.parameters()))

for p in gpt.parameters():
    p.requires_grad = False

block_size = config.block_size # context length is a property of the model, so read it from the config

# encode Shakespeare with the checkpoint's vocab, so each id means the same character as in pretraining
tok = CharTokenizer(ckpt['chars'])
with open("nanogpt/input/input.txt", "r", encoding="utf-8") as file:
    text = file.read()
data = torch.tensor(tok.encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

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

def save_lora(lr_mat, step, best_val):
    checkpoint = {
        'lora': lr_mat,
        'r': r,
        'alpha': alpha,
        'step': step,
        'best_val': best_val,
        }
    torch.save(checkpoint, 'nanogpt/output/shakespeare_lora.pt')
    print(f"saved checkpoint at step {step} (val {best_val:.4f})")

apply_lora(gpt, r, alpha)

print(sum(p.numel() for p in gpt.parameters() if p.requires_grad))

lora_params = [p for p in gpt.parameters() if p.requires_grad]

optimizer = torch.optim.AdamW(lora_params, lr=1e-3)
best_val = float('inf')

gpt.train()
for steps in range(max_iters):
  xb, yb = get_batch('train')

  logits, loss = gpt(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

  if (steps % eval_interval == 0) or (steps == max_iters - 1):
    losses = estimate_loss()
    print(steps, losses['train'], losses['val'])

    if losses['val'] < best_val:
        best_val = losses['val']

        lora_state = {k: v for k, v in gpt.state_dict().items() if k.endswith(('.A', '.B'))}
        save_lora(lora_state, steps, best_val)




