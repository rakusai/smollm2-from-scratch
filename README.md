# smollm2-from-scratch

A from-scratch PyTorch implementation of [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) (Llama architecture). It loads the official pretrained weights and generates text. The only thing borrowed from Hugging Face at runtime is the tokenizer.

The whole model is about 200 lines in [`model.py`](model.py):

| Part | Details |
|---|---|
| Embedding | 49,152 vocab × 576 dims, shared with the output layer |
| Decoder layer ×30 | RMSNorm → attention → residual → RMSNorm → SwiGLU MLP (1536) → residual |
| Attention | Grouped-query attention (9 query heads, 3 KV heads) with RoPE (θ = 100,000) |
| Generation | KV cache (prefill, then one token at a time), temperature + top-k sampling |

Its output matches the `transformers` reference implementation exactly: the max logit difference is 0.0 for a full prompt, and greedy generation produces the same tokens.

For an illustrated walkthrough of the files, the architecture, and how a prompt becomes text, see **[SmolLM2 Anatomy](https://rakusai.github.io/smollm2-from-scratch/smollm2-anatomy.html)** ([Japanese version](https://rakusai.github.io/smollm2-from-scratch/smollm2-anatomy-ja.html)). The source is in [`docs/`](docs/).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Model weights

You don't need to download anything by hand. On the first run, the scripts fetch the weights (~270 MB) from the Hugging Face Hub and cache them under `~/.cache/huggingface/hub/`. Later runs reuse the cache. To skip the network check once the files are cached, set `HF_HUB_OFFLINE=1`.

## Usage

Generate text:

```bash
.venv/bin/python generate.py "Once upon a time, in a small village," --max-new-tokens 100
```

Options: `--temperature` (0 = greedy), `--top-k`, `--seed`, `--device` (`cpu` by default; `mps` and `cuda` also work).

Check against the reference implementation (requires `transformers`):

```bash
.venv/bin/python verify.py
```

## Files

| File | Purpose |
|---|---|
| `model.py` | Model definition and weight loading |
| `generate.py` | Command-line text generation |
| `verify.py` | Comparison with Hugging Face `transformers` |
| `docs/smollm2-anatomy.html` | Illustrated explanation |
| `docs/smollm2-anatomy-ja.html` | Illustrated explanation (Japanese) |

## Performance

On an Apple M4 (CPU), generation runs at about 70 tokens/s. Each token takes about 14 ms: MLP 40%, attention 38%, output layer 13%. Speed is limited mainly by reading the ~540 MB of float32 weights from memory for every token.

## License

The code in this repository is released under the [MIT License](LICENSE). The SmolLM2 weights are not included here; they are distributed by Hugging Face under the Apache 2.0 license.
