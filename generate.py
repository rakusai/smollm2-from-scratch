"""Generate text with the from-scratch SmolLM2.

Usage:
  .venv/bin/python generate.py "Once upon a time" --max-new-tokens 100
"""

import argparse
import sys
import time

import torch
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from model import SmolLM2

REPO = "HuggingFaceTB/SmolLM2-135M"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="The meaning of life is")
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")  # at 135M params, CPU is faster than MPS
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model_dir = snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors"])
    tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
    model = SmolLM2.from_pretrained(model_dir).to(args.device)

    ids = tok.encode(args.prompt).ids
    idx = torch.tensor([ids], device=args.device)

    print(args.prompt, end="", flush=True)
    out, printed = [], ""
    t0 = time.time()
    for t in model.generate(idx, args.max_new_tokens, args.temperature, args.top_k, eos_id=0):
        out.append(t)
        text = tok.decode(out)  # decode everything and print the diff so multi-byte chars are never split
        sys.stdout.write(text[len(printed):])
        sys.stdout.flush()
        printed = text
    dt = time.time() - t0
    print(f"\n\n[{len(out)} tokens, {len(out) / dt:.1f} tok/s on {args.device}]")


if __name__ == "__main__":
    main()
