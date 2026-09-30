"""Check that our implementation matches the Hugging Face transformers reference."""

import torch
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

from model import SmolLM2

REPO = "HuggingFaceTB/SmolLM2-135M"

model_dir = snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors"])
tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
ids = torch.tensor([tok.encode("The capital of France is Paris, and the capital of Japan is").ids])

ours = SmolLM2.from_pretrained(model_dir)
ref = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()

with torch.no_grad():
    a = ours(ids)
    b = ref(ids).logits
print("max |logit diff| (full sequence):", (a - b).abs().max().item())

# same result when feeding one token at a time through the KV cache?
with torch.no_grad():
    caches = [[] for _ in ours.layers]
    steps = [ours(ids[:, i:i + 1], i, caches) for i in range(ids.shape[1])]
    c = torch.cat(steps, dim=1)
print("max |logit diff| (KV cache):", (c - b).abs().max().item())

# compare greedy generation
g_ours = list(ours.generate(ids, 20, temperature=0))
g_ref = ref.generate(ids, max_new_tokens=20, do_sample=False)[0, ids.shape[1]:].tolist()
print("ours:", repr(tok.decode(g_ours)))
print("ref :", repr(tok.decode(g_ref)))
assert g_ours == g_ref, "greedy outputs differ"
print("OK: matches the reference implementation")
