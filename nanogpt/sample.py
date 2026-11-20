import torch
from model import GPTConfig, GPTLanguageModel, apply_lora
from tokenizer import CharTokenizer

# settings
ckpt_path = 'nanogpt/output/ckpt.pt'
out_path = 'nanogpt/output/output_lora.txt'
lora_path = 'nanogpt/output/shakespeare_lora.pt'
max_new_tokens = 1000
seed = 1337
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
# ------------

torch.manual_seed(seed)

# everything needed to rebuild the model and tokenizer lives in the checkpoint
ckpt = torch.load(ckpt_path, map_location=device)
gpt = GPTLanguageModel(GPTConfig(**ckpt['config']))
gpt.load_state_dict(ckpt['model'])

if lora_path:
    lora = torch.load(lora_path, map_location=device)
    apply_lora(gpt, lora['r'], lora['alpha'])
    gpt.load_state_dict(lora['lora'], strict=False)


gpt.to(device)

tok = CharTokenizer(ckpt['chars'])

def generate_output(model, new_tokens, context = '\n'):
    model.eval()
    idx = torch.tensor([tok.encode(context)], dtype=torch.long, device=device)
    out = tok.decode(model.generate(idx, max_new_tokens=new_tokens)[0])
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(out)
    print(f"wrote {len(out)} characters to {out_path}")

generate_output(gpt, max_new_tokens, 'First Citizen: Before we proceed any further, hear me speak.')
