"""
A minimal, fast example generating text with Llama 3.1 in MLX.

To run, install the requirements:

    pip install -U mlx transformers fire

Then generate text with:

    python l3min.py "How tall is K2?"
"""

import fire
import json
import glob
from huggingface_hub import snapshot_download
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
import time
from transformers import AutoTokenizer
from types import SimpleNamespace


class DynamicNTKScalingRoPE(nn.Module):

    def __init__(
        self,
        dims,
        rope_scaling,
        max_position_embeddings=2048,
        base=10000,
    ):
        super().__init__()
        self.dims = dims
        self.max_position_embeddings = max_position_embeddings

        factor = rope_scaling["factor"]
        low_freq_factor = rope_scaling["low_freq_factor"]
        high_freq_factor = rope_scaling["high_freq_factor"]
        old_context_len = rope_scaling["original_max_position_embeddings"]

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        freqs = base ** (mx.arange(0, self.dims, 2) / self.dims)
        wavelens = 2 * mx.pi * freqs

        freqs = mx.where(wavelens > low_freq_wavelen, freqs * factor, freqs)
        is_medium_freq = (wavelens > high_freq_wavelen) & (wavelens < low_freq_wavelen)
        smooth_factors = (old_context_len / wavelens - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smooth_freqs = freqs / ((1 - smooth_factors) / factor + smooth_factors)
        self._freqs = mx.where(is_medium_freq, smooth_freqs, freqs)

    def __call__(self, x, offset=0):
        return mx.fast.rope(
            x,
            self.dims,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.hidden_size // n_heads
        self.scale = head_dim ** (-0.5)

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope = DynamicNTKScalingRoPE(
            dims=head_dim,
            rope_scaling=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
            base=args.rope_theta,
        )

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, mask=mask, scale=self.scale
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), (keys, values)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x, mask=None, cache=None):
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out, cache


class LlamaModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache=None):
        h = self.embed_tokens(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        return self.norm(h), cache


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = LlamaModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        out, cache = self.model(inputs, cache)
        return self.lm_head(out), cache


def load(hf_repo):
    model_path = Path(
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=["*.json", "*.safetensors"],
        )
    )
    with open(model_path / "config.json", "r") as f:
        config = json.load(f)

    weight_files = glob.glob(str(model_path / "model*.safetensors"))
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    model = Model(SimpleNamespace(**config))

    if (quantization := config.get("quantization", None)) is not None:
        nn.quantize(model, **quantization)

    model.load_weights(list(weights.items()))

    mx.eval(model.parameters())

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.decode([0])

    return model, tokenizer


def generate_step(prompt, model):
    cache = None

    def _step(y):
        nonlocal cache
        logits, cache = model(y, cache=cache)
        return mx.argmax(logits[:, -1, :], axis=-1)

    y = _step(prompt)
    mx.async_eval(y)
    while True:
        next_y = _step(y[None])
        mx.async_eval(next_y)
        yield y.item()
        y = next_y


def generate(
    prompt,
    model="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    max_tokens=128,
):
    print("[INFO] Loading model from disk.")
    model, tokenizer = load(model)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="mlx",
    )

    print("[INFO] Starting generation...")
    tic = time.time()
    s = 0
    tokens = []
    for token, n in zip(generate_step(prompt, model), range(max_tokens)):
        tokens.append(token)
        if n == 0:
            prompt_tps = prompt.size / (time.time() - tic)
            tic = time.time()

        if token == tokenizer.eos_token_id:
            break

        words = tokenizer.decode(tokens)
        print(words[s:], end="", flush=True)
        if words[-1] == "\n":
            tokens = []
            s = 0
        else:
            s = len(words)

    print(tokenizer.decode(tokens)[s:], flush=True)
    gen_tps = (n + 1) / (time.time() - tic)
    print("=" * 10)
    print(f"Prompt: {prompt_tps:.3f} tokens-per-sec")
    print(f"Generation: {gen_tps:.3f} tokens-per-sec")


if __name__ == "__main__":
    fire.Fire(generate)
