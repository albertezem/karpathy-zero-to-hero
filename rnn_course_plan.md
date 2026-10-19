# makemore Part RNN: Recurrent Neural Networks, from scratch

*The lecture Karpathy never recorded, done in his style. Every step uses raw tensors and `@`, with no `nn.RNN` until we've earned it.*

**Where we are:** the MLP from makemore pt2/pt3. It reads `block_size = 3` characters, embeds them with `C`, concatenates them, sends them through a tanh hidden layer (with batchnorm and Kaiming init) and outputs logits. Val loss is about 2.1.

**The problem with it:** the MLP only ever sees 3 characters. To see more you raise `block_size`, but then `W1` grows linearly, and each position gets its *own* weights. The network can't reuse what it learned about "an" at position 1 when "an" shows up at position 5. There's also a hard limit: nothing beyond the window can ever affect the prediction.

**The idea of this lecture:** keep a fixed-size vector `h` that summarizes *everything seen so far*. Update it one character at a time, and use **the same weights** at every step.

```
h_t = tanh( x_t @ Wxh + h_{t-1} @ Whh + bh )
logits_t = h_t @ Why + by
```

That's the whole RNN, and the rest of the lecture works through what follows from it.

---

## Part 1: The whole RNN on one name at a time (no batching, no padding)
- For one word: `xs = [0] + chars`, `ys = chars + [0]`. That's the whole "dataset" for one example.
- Parameters: `C`, `Wxh`, `Whh`, `bh`, `Why`, `by`. Write down every shape before you create anything.
- Forward pass: a Python `for` loop over the characters of `"emma"`, updating `h` and producing logits at every step. Average the per-step losses.
- ✅ Loss at init should be about `-ln(1/27) ≈ 3.29`. If it isn't, fix the init (pt3 lessons).
- Backward, update, and loop over the names one at a time (batch size 1). Watch the loss go down.
- Sample: feed one char at a time until the model outputs `.`. You get variable length for free.
- 💡 The same `Whh` is used at every step. That's **weight sharing across time**, the same idea a conv net uses across space.
- 💡 Notice how *slow* it is: one tiny matmul per character, in a Python loop. That's the motivation for Part 2.

## Part 2: Batching needs padding: the (N, T) dataset
- To do 32 names in one matmul, the tensor must be rectangular, so we pad every name to `T = max_len + 1 = 16`.
- Build `X` and `Y` of shape `(N, T)`. `Y` is `X` shifted left by one.
- Pad `X` with `0` (any valid index works, since outputs there are thrown away). Pad `Y` with `-1` so the loss **ignores** it (`ignore_index=-1`).
- ✅ Check: decode a few rows of X and Y side by side and make sure they line up.
- 💡 *Seed for later:* "every position is a training example, and targets are the inputs shifted by one" is **exactly** how GPT is trained.

## Part 3: The batched forward pass
- `emb = C[Xb]` → `(B, T, n_embd)`. Loop over `t`, collect the `h_t`s, and `torch.stack` them into `(B, T, n_hidden)`.
- Logits `(B, T, vocab)` → reshape to `(B*T, vocab)` for `F.cross_entropy(..., ignore_index=-1)`.
- ✅ Check: the batched forward on a batch of 1 gives the same loss as your Part 1 loop.
- Exercise: show that `torch.cat([x, h]) @ W` equals `x @ Wxh + h @ Whh` when `W` stacks `Wxh` on top of `Whh`. They're the same thing.

## Part 4: Train and sample
- Use the same training loop as pt3: minibatches, lr decay, `lossi` plot, and `split_loss` on train/val.
- Sampling gets *nicer*: carry `h` forward and feed one char at a time. There's no context window to crop.
- ✅ Compare val loss against your MLP. Does unbounded context help? (Expect roughly the 2.0 to 2.1 region, and note by how much.)

## Part 5: Diagnostics: backprop through time and vanishing/exploding gradients
- Call `retain_grad()` on each `h_t`. Plot `‖dL/dh_t‖` against `t`. What happens to gradient flowing back from late positions to early ones?
- Connect this to pt3: every step back through time multiplies by `Whh` and by `(1 - h²)`, so gradients shrink or blow up geometrically. Look at the singular values of `Whh`.
- Experiment: scale the `Whh` init up and down and watch it. Add **gradient clipping** (`torch.nn.utils.clip_grad_norm_`).

## Part 6: Check against PyTorch
- Copy your weights into `torch.nn.RNN` and confirm that your hidden states match to about 1e-6. Now you know what the library does.
- Wrap your code in a small class and compare speed.

## Part 7: Gating: the GRU
- Fix vanishing gradients by making the update **additive and gated**:
  `h_t = (1 - z) * h_{t-1} + z * h̃_t`
  This gives the gradient a "highway" through time.
- Implement a GRU cell from scratch (reset gate `r`, update gate `z`, candidate `h̃`). Train it and compare with the vanilla RNN.
- Redo the Part 5 gradient-norm plot and compare.
- Mention the LSTM (cell state `c` plus 3 gates) as a close sibling.
- 💡 An additive path for gradients shows up again as **residual connections** in ResNets and in the transformer's "residual stream".

## Part 8: What went wrong, and the road to GPT
- **Bottleneck:** the entire past is squeezed into one `n_hidden` vector. *Attention* (Bahdanau 2014) was invented to let an RNN look back at all past states directly, not just the last one.
- **Sequential:** `h_t` depends on `h_{t-1}`, so training can't be parallelized over `T`. Transformers drop the recurrence, process all positions at once and add a causal mask.
- **Kept:** the dataset from Part 2, next-token prediction at every position, residual/additive paths, and embeddings.
- **Aside:** RNNs are coming back in linear-recurrence form (RWKV, Mamba/SSMs), which keeps training parallel and generation cheap.
- **Reading:** Karpathy's *The Unreasonable Effectiveness of Recurrent Neural Networks* (2015), and `min-char-rnn.py`.

## Exercises (take-home, Karpathy style)
1. **Backprop ninja, through time:** write the RNN backward pass by hand and check it against autograd.
2. Implement an LSTM cell from scratch and try to beat the GRU's val loss.
3. Tune `n_embd`, `n_hidden`, init and lr. How low can you get val loss?
4. Stack 2 RNN layers, so the `h`s of layer 1 become the inputs to layer 2.
5. Train on a different text (for example tiny Shakespeare) and sample from it.
