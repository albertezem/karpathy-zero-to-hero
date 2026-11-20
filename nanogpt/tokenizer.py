import torch

class CharTokenizer:
    # one token per character; the order of chars defines the token ids,
    # so a model must always be paired with the chars list it was trained with
    def __init__(self, chars):
        self.chars = list(chars)
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return ''.join([self.itos[i] for i in ids])
