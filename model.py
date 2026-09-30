"""SmolLM2-135M (Llama architecture) implemented in plain PyTorch.

Components:
  - Token embedding (shared with the output layer = tie_word_embeddings)
  - Decoder block x 30
      RMSNorm -> Grouped-Query Attention (RoPE) -> residual
      RMSNorm -> SwiGLU MLP -> residual
  - Final RMSNorm -> LM head
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    rope_theta: float = 100000.0
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_json(cls, path) -> "Config":
        raw = json.loads(Path(path).read_text())
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})


class RMSNorm(nn.Module):
    """x / sqrt(mean(x^2) + eps) * weight. LayerNorm without mean subtraction and bias."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def rope_cache(cfg: Config, device):
    """Precompute cos/sin for every position and frequency. shape: (max_pos, head_dim)"""
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, device=device).float() / cfg.head_dim))
    t = torch.arange(cfg.max_position_embeddings, device=device).float()
    freqs = torch.outer(t, inv_freq)            # (max_pos, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)     # same frequencies in both halves (non-interleaved)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    # x: (B, heads, T, head_dim), cos/sin: (T, head_dim)
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    """Grouped-Query Attention: 9 query heads, 3 key/value heads; every 3 query heads share one K/V head."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.hd = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.hd, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin, kv_cache=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            if kv_cache:  # append to the K/V of past tokens
                k = torch.cat([kv_cache[0], k], dim=2)
                v = torch.cat([kv_cache[1], v], dim=2)
            kv_cache[:] = [k, v]

        # repeat K/V heads to match the number of query heads
        rep = self.n_heads // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        # causal mask for the prompt (T>1); not needed when decoding one token at a time
        out = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    """SwiGLU: down( silu(gate(x)) * up(x) )"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, kv_cache=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, kv_cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class SmolLM2(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        cos, sin = rope_cache(cfg, device="cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, idx, start_pos: int = 0, kv_caches=None):
        T = idx.shape[1]
        cos = self.rope_cos[start_pos:start_pos + T].to(self.embed_tokens.weight.dtype)
        sin = self.rope_sin[start_pos:start_pos + T].to(self.embed_tokens.weight.dtype)
        x = self.embed_tokens(idx)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, kv_caches[i] if kv_caches is not None else None)
        return self.lm_head(self.norm(x))

    @classmethod
    def from_pretrained(cls, model_dir, dtype=torch.float32):
        """Load Hugging Face safetensors. Stripping the 'model.' prefix is all the key mapping needed."""
        from safetensors.torch import load_file

        model_dir = Path(model_dir)
        cfg = Config.from_json(model_dir / "config.json")
        model = cls(cfg)
        sd = load_file(model_dir / "model.safetensors")
        sd = {k.removeprefix("model."): v for k, v in sd.items()}
        if cfg.tie_word_embeddings:
            sd.setdefault("lm_head.weight", sd["embed_tokens.weight"])
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        return model.to(dtype).eval()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, eos_id=None):
        kv_caches = [[] for _ in self.layers]
        logits = self(idx, 0, kv_caches)[:, -1]  # process the whole prompt at once (prefill)
        pos = idx.shape[1]
        for _ in range(max_new_tokens):
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -math.inf
                nxt = torch.multinomial(F.softmax(logits.float(), -1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            yield nxt.item()
            if eos_id is not None and nxt.item() == eos_id:
                break
            logits = self(nxt, pos, kv_caches)[:, -1]  # process only the new token (decode)
            pos += 1
