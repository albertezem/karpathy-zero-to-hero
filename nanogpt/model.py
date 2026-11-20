import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 96
    block_size: int = 256
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.0

class Head(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.head_size = head_size
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size))) # not optimized but part of Head

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B,T, head_size)
        q = self.query(x) # B,T, head_size
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5

        tril = self.tril[:T, :T]

        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)

        out = wei @ v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        outs = []
        for head in self.heads:
            outs.append(head(x))

        outs = torch.cat(outs, dim=-1)

        return self.dropout(self.proj(outs))

class CausalSelfAttention(nn.Module): # replaces multiheadattention 
    def __init__(self, config) -> None:
        super().__init__()
        n_head = config.n_head
        head_size = config.n_embd // n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * head_size * n_head, bias=False) # idea: 1 matrix that contains all K/Q/V values for all the heads
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.n_head = n_head
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1,2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1,2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1,2)

        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        
        tril = self.tril[:T, :T]

        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)

        out = wei @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.proj(out))
        
        

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        ) 

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = CausalSelfAttention(config)
        self.ff = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.heads(self.ln1(x))
        x = x + self.ff(self.ln2(x))

        return x

class GPTLanguageModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embeddings = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size) 


    def forward(self, idx, targets=None):

        B, T = idx.shape 
        tok_emb = self.token_embeddings(idx)
        pos_emb = self.pos_embeddings(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx[:, -self.config.block_size:])
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha) -> None:
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn((r, base.in_features), device=base.weight.device) / base.in_features**0.5)
        self.B = nn.Parameter(torch.zeros((base.out_features, r), device=base.weight.device))

    def forward(self, x):
        delta = (x @ self.A.T) @ self.B.T * self.scale
        out = self.base(x) + delta

        return out
    
def apply_lora(model, r, alpha):
    for block in model.blocks:
        block.heads.c_attn = LoRALinear(block.heads.c_attn, r, alpha)
        block.heads.proj = LoRALinear(block.heads.proj, r, alpha)
