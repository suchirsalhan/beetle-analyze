# =========================================================
# INSTALLS
# =========================================================
#!pip install -q -U huggingface_hub transformers torch tokenizers

# =========================================================
# IMPORTS
# =========================================================
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
import os, re, json, tempfile, time
from collections import defaultdict

api = HfApi()
print("✅ Using HuggingFace API")

# =========================================================
# MODEL LIST
# =========================================================
ALL_REPOS = [
    "BeetleLM/beetlelm_zho_L1-nld_L2-eng_L3_part_time",
    "BeetleLM/beetlelm_zho_L1-eng_L2-nld_L3_part_time",
    "BeetleLM/beetlelm_nld_L1-zho_L2-eng_L3_part_time",
    "BeetleLM/beetlelm_nld_L1-eng_L2-zho_L3_part_time",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_part_time",
    "BeetleLM/beetlelm_eng_L1-zho_L2-nld_L3_part_time",

    "BeetleLM/beetlelm_zho_L1-nld_L2-eng_L3_late",
    "BeetleLM/beetlelm_zho_L1-eng_L2-nld_L3_late",
    "BeetleLM/beetlelm_nld_L1-zho_L2-eng_L3_late",
    "BeetleLM/beetlelm_nld_L1-eng_L2-zho_L3_late",
    "BeetleLM/beetlelm_eng_L1-zho_L2-nld_L3_late",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_late",

    "BeetleLM/beetlelm_zho_L1-nld_L2-eng_L3_heritage",
    "BeetleLM/beetlelm_zho_L1-eng_L2-nld_L3_heritage",
    "BeetleLM/beetlelm_nld_L1-zho_L2-eng_L3_heritage",
    "BeetleLM/beetlelm_nld_L1-eng_L2-zho_L3_heritage",
    "BeetleLM/beetlelm_eng_L1-zho_L2-nld_L3_heritage",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_heritage",

    "BeetleLM/beetlelm_zho_L1-eng_L2-nld_L3_balanced",
    "BeetleLM/beetlelm_zho_L1-nld_L2-eng_L3_balanced",
    "BeetleLM/beetlelm_nld_L1-zho_L2-eng_L3_balanced",
    "BeetleLM/beetlelm_nld_L1-eng_L2-zho_L3_balanced",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_balanced",
    "BeetleLM/beetlelm_eng_L1-zho_L2-nld_L3_balanced",

    "BeetleLM/beetlelm_ukr_mono",
    "BeetleLM/beetlelm_ind_mono",
    "BeetleLM/beetlelm_fra_mono",
    "BeetleLM/beetlelm_fas_mono",
    "BeetleLM/beetlelm_deu_mono",
    "BeetleLM/beetlelm_bul_mono",

    "BeetleLM/beetlelm_zho-ukr_simultaneous",
    "BeetleLM/beetlelm_nld-ukr_simultaneous",
    "BeetleLM/beetlelm_zho-fas_simultaneous",
    "BeetleLM/beetlelm_fra-ind_simultaneous",
    "BeetleLM/beetlelm_nld-bul_simultaneous",
    "BeetleLM/beetlelm_zho-ind_simultaneous",
    "BeetleLM/beetlelm_fas-ind_simultaneous",
    "BeetleLM/beetlelm_fas-bul_simultaneous",
    "BeetleLM/beetlelm_zho-deu_simultaneous",
    "BeetleLM/beetlelm_nld-fra_simultaneous",
    "BeetleLM/beetlelm_bul-ukr_simultaneous",
    "BeetleLM/beetlelm_eng-fra_simultaneous",
    "BeetleLM/beetlelm_ind-deu_simultaneous",
    "BeetleLM/beetlelm_fas-ukr_simultaneous",
    "BeetleLM/beetlelm_zho-bul_simultaneous",
    "BeetleLM/beetlelm_zho-fra_simultaneous",
    "BeetleLM/beetlelm_fas-eng_simultaneous",
    "BeetleLM/beetlelm_nld-deu_simultaneous",
    "BeetleLM/beetlelm_fra-ukr_simultaneous",

    "BeetleLM/beetlelm_zho-fas_sequential",
    "BeetleLM/beetlelm_zho-ukr_sequential",
    "BeetleLM/beetlelm_zho-ind_sequential",
    "BeetleLM/beetlelm_zho-deu_sequential",
    "BeetleLM/beetlelm_nld-bul_sequential",
    "BeetleLM/beetlelm_nld-ukr_sequential",
    "BeetleLM/beetlelm_zho-fra_sequential",
    "BeetleLM/beetlelm_fas-ind_sequential",
    "BeetleLM/beetlelm_fra-ind_sequential",
    "BeetleLM/beetlelm_ind-deu_sequential",
    "BeetleLM/beetlelm_nld-fra_sequential",
    "BeetleLM/beetlelm_fas-bul_sequential",
    "BeetleLM/beetlelm_eng-fra_sequential",
    "BeetleLM/beetlelm_fas-ukr_sequential",
    "BeetleLM/beetlelm_fas-eng_sequential",
    "BeetleLM/beetlelm_nld-deu_sequential",
    "BeetleLM/beetlelm_zho-bul_sequential",
    "BeetleLM/beetlelm_eng-nld_sequential",
    "BeetleLM/beetlelm_eng-bul_sequential",

    "BeetleLM/beetlelm_zho-ukr_part_time",
    "BeetleLM/beetlelm_zho-ind_part_time",
    "BeetleLM/beetlelm_nld-ukr_part_time",
    "BeetleLM/beetlelm_zho-fas_part_time",
    "BeetleLM/beetlelm_nld-fra_part_time",
    "BeetleLM/beetlelm_zho-deu_part_time",
    "BeetleLM/beetlelm_fra-ind_part_time",
    "BeetleLM/beetlelm_nld-bul_part_time",
    "BeetleLM/beetlelm_ind-deu_part_time",
    "BeetleLM/beetlelm_fas-ukr_part_time",
    "BeetleLM/beetlelm_zho-bul_part_time",
    "BeetleLM/beetlelm_zho-fra_part_time",
    "BeetleLM/beetlelm_fas-ind_part_time",
    "BeetleLM/beetlelm_fas-bul_part_time",
    "BeetleLM/beetlelm_fas-eng_part_time",
    "BeetleLM/beetlelm_eng-nld_part_time",
    "BeetleLM/beetlelm_fra-ukr_part_time",
    "BeetleLM/beetlelm_nld-deu_part_time",
    "BeetleLM/beetlelm_eng-fra_part_time",
    "BeetleLM/beetlelm_bul-ukr_part_time",
    "BeetleLM/beetlelm_fas-nld_part_time",
    "BeetleLM/beetlelm_fas-deu_part_time",
    "BeetleLM/beetlelm_bul-fra_part_time",
    "BeetleLM/beetlelm_eng-bul_part_time",
    "BeetleLM/beetlelm_deu-ukr_part_time",
    "BeetleLM/beetlelm_eng-ind_part_time",
]

# =========================================================
# WRITE pico_decoder.py
# =========================================================
# ── Write pico_decoder.py ─────────────────────────────────────────────────────
PICO_PATH = "/content/pico_decoder.py"

with open(PICO_PATH, "w") as _f:
    _f.write('''\
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput, CausalLMOutputWithPast
try:
    if TYPE_CHECKING:
        from src.config import ModelConfig
except ImportError:
    pass


class RMSNorm(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps    = config.norm_eps
        self.weight = nn.Parameter(torch.ones(config.d_model))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class RoPE(nn.Module):
    """
    Rotary Position Embedding.
    freqs_cis is computed lazily on first use and cached per-device,
    avoiding meta-tensor issues when HF loads with low_cpu_mem_usage=True.
    """
    def __init__(self, config):
        super().__init__()
        self.theta   = config.position_emb_theta
        self.dim     = config.d_model // config.attention_n_heads
        self.max_seq = config.max_seq_len
        # NOT a buffer — plain dict so it never touches the meta device
        self._cache: Dict[torch.device, torch.Tensor] = {}

    def _get_freqs_cis(self, device: torch.device) -> torch.Tensor:
        if device not in self._cache:
            freqs = 1.0 / (
                self.theta ** (
                    torch.arange(0, self.dim, 2, device=device).float() / self.dim
                )
            )
            t = torch.arange(self.max_seq, device=device)
            freqs = torch.outer(t, freqs)
            self._cache[device] = torch.polar(torch.ones_like(freqs), freqs)
        return self._cache[device]

    def get_freqs_cis(self, input_shape, start_pos, end_pos, device):
        _f   = self._get_freqs_cis(device)[start_pos:end_pos]
        ndim = len(input_shape)
        assert 0 <= 1 < ndim and _f.shape == (input_shape[1], input_shape[-1])
        return _f.view(*[d if i == 1 or i == ndim - 1 else 1
                         for i, d in enumerate(input_shape)])

    def forward(self, queries, keys, start_pos=0):
        device = queries.device
        q_ = torch.view_as_complex(queries.float().reshape(*queries.shape[:-1], -1, 2))
        k_ = torch.view_as_complex(keys.float().reshape(*keys.shape[:-1], -1, 2))
        fc  = self.get_freqs_cis(q_.shape, start_pos, start_pos + q_.shape[1], device)
        return (torch.view_as_real(q_ * fc).flatten(3).type_as(queries),
                torch.view_as_real(k_ * fc).flatten(3).type_as(keys))


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads     = config.attention_n_heads
        self.n_kv_heads  = config.attention_n_kv_heads
        self.batch_size  = config.batch_size
        self.max_seq_len = config.max_seq_len
        d = config.d_model
        self.head_dim = d // self.n_heads
        self.n_rep    = self.n_heads // self.n_kv_heads
        self.q_proj = nn.Linear(d, self.n_heads    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d,    bias=False)
        self.rope   = RoPE(config)
    def forward(self, input, mask=None, past_key_values=None, use_cache=False):
        bsz, seq_len, _ = input.shape
        queries = self.q_proj(input).view(bsz, seq_len, self.n_heads,    self.head_dim)
        keys    = self.k_proj(input).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        values  = self.v_proj(input).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        start_pos = past_key_values[0].shape[1] if past_key_values is not None else 0
        queries, keys = self.rope(queries, keys, start_pos)
        if past_key_values is not None:
            keys   = torch.cat([past_key_values[0], keys],   dim=1)
            values = torch.cat([past_key_values[1], values], dim=1)
        cached_keys   = keys   if use_cache else None
        cached_values = values if use_cache else None
        queries = queries.transpose(1, 2)
        keys    = keys.transpose(1, 2)
        values  = values.transpose(1, 2)
        apply_gqa = self.n_rep > 1
        if apply_gqa and queries.device.type == "mps":
            keys   = keys.repeat_interleave(self.n_rep, dim=-3)
            values = values.repeat_interleave(self.n_rep, dim=-3)
            apply_gqa = False
        attn_mask = mask.to(queries.dtype) if mask is not None else None
        with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]):
            attn_output = F.scaled_dot_product_attention(
                queries.contiguous(), keys.contiguous(), values.contiguous(),
                attn_mask=attn_mask, enable_gqa=apply_gqa,
            )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output), (cached_keys, cached_values)


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w_0 = nn.Linear(config.d_model, config.activation_hidden_dim, bias=False)
        self.w_1 = nn.Linear(config.d_model, config.activation_hidden_dim, bias=False)
        self.w_2 = nn.Linear(config.activation_hidden_dim, config.d_model, bias=False)
    def forward(self, x):
        return self.w_2(F.silu(self.w_0(x)) * self.w_1(x))


class PicoDecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention      = Attention(config)
        self.swiglu         = SwiGLU(config)
        self.attention_norm = RMSNorm(config)
        self.swiglu_norm    = RMSNorm(config)
    def forward(self, input, mask=None, past_key_values=None, use_cache=False):
        attention_output, cached_key_values = self.attention(
            self.attention_norm(input), mask=mask,
            past_key_values=past_key_values, use_cache=use_cache)
        h = input + attention_output
        return h + self.swiglu(self.swiglu_norm(h)), cached_key_values


class PicoDecoder(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.config            = model_config
        self.embedding_proj    = nn.Embedding(model_config.vocab_size, model_config.d_model)
        self.layers            = nn.ModuleList(
            [PicoDecoderBlock(model_config) for _ in range(model_config.n_layers)])
        self.output_norm       = RMSNorm(model_config)
        self.de_embedding_proj = nn.Linear(
            model_config.d_model, model_config.vocab_size, bias=False)
    def convert_to_hf_model(self):
        hf = PicoDecoderHF(PicoDecoderHFConfig.from_dataclass(self.config))
        hf.load_state_dict(self.state_dict())
        return hf
    def forward(self, input_ids, past_key_values=None, use_cache=False):
        seq_len   = input_ids.shape[-1]
        h         = self.embedding_proj(input_ids)
        start_pos = 0 if past_key_values is None else past_key_values[0][0].shape[1]
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            if past_key_values is not None:
                mask = torch.hstack([torch.zeros((seq_len, start_pos)), mask])
            mask = mask.to(h.device)
        cached_key_values = () if use_cache else None
        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            h, layer_cached = layer(
                h, mask=mask, past_key_values=layer_past, use_cache=use_cache)
            if use_cache:
                cached_key_values += (layer_cached,)
        return self.de_embedding_proj(self.output_norm(h)).float(), cached_key_values


class PicoDecoderHFConfig(PretrainedConfig):
    model_type = "pico_decoder"
    def __init__(self,
                 n_layers=14, d_model=768, vocab_size=32768,
                 attention_n_heads=12, attention_n_kv_heads=1,
                 max_seq_len=512, batch_size=64, position_emb_theta=10000.0,
                 activation_hidden_dim=3072, norm_eps=1e-5, dropout=0.1,
                 **kwargs):
        if not attention_n_kv_heads:
            attention_n_kv_heads = attention_n_heads
        super().__init__(**kwargs)
        self.n_layers              = n_layers
        self.d_model               = d_model
        self.vocab_size            = vocab_size
        self.attention_n_heads     = attention_n_heads
        self.attention_n_kv_heads  = attention_n_kv_heads
        self.max_seq_len           = max_seq_len
        self.batch_size            = batch_size
        self.position_emb_theta    = position_emb_theta
        self.activation_hidden_dim = activation_hidden_dim
        self.norm_eps              = norm_eps
        self.dropout               = dropout
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PicoDecoderHFConfig":
        pico_config = cls(**config_dict)
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        unused_kwargs = {k: v for k, v in kwargs.items() if not hasattr(pico_config, k)}
        if return_unused_kwargs:
            return pico_config, unused_kwargs
        return pico_config
    @classmethod
    def from_dataclass(cls, model_config):
        return cls.from_dict(asdict(model_config))


class PicoDecoderHF(PreTrainedModel):
    """
    HuggingFace wrapper for BeetleLM PicoDecoder.
    Usage: AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
    Works with CPU, CUDA (A100, etc.), and MPS out of the box.
    """
    config_class       = PicoDecoderHFConfig
    _no_split_modules  = ["PicoDecoderBlock"]
    _tied_weights_keys = []

    def __init__(self, config: PicoDecoderHFConfig):
        super().__init__(config)
        self.embedding_proj    = nn.Embedding(config.vocab_size, config.d_model)
        self.layers            = nn.ModuleList(
            [PicoDecoderBlock(config) for _ in range(config.n_layers)])
        self.output_norm       = RMSNorm(config)
        self.de_embedding_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Required: lets HF finalize weight init and meta-device materialization
        self.post_init()

    # Required for low_cpu_mem_usage / Accelerate device-dispatch to work
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def get_input_embeddings(self):        return self.embedding_proj
    def set_input_embeddings(self, value): self.embedding_proj = value

    def forward(self, input_ids=None, past_key_values=None,
                use_cache=False, labels=None, **kwargs):
        seq_len   = input_ids.shape[-1]
        h         = self.embedding_proj(input_ids)
        start_pos = 0 if past_key_values is None else past_key_values[0][0].shape[1]
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=1)
            if past_key_values is not None:
                mask = torch.hstack([torch.zeros((seq_len, start_pos), device=h.device), mask])
        cached_key_values = () if use_cache else None
        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            h, layer_cached = layer(
                h, mask=mask, past_key_values=layer_past, use_cache=use_cache)
            if use_cache:
                cached_key_values += (layer_cached,)
        logits = self.de_embedding_proj(self.output_norm(h)).float()
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().clamp(0, self.config.vocab_size - 1).view(-1),
            )
        if use_cache:
            return CausalLMOutputWithPast(
                loss=loss, logits=logits, past_key_values=cached_key_values)
        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        return {"input_ids": input_ids,
                "past_key_values": past_key_values,
                "use_cache": True}


PicoDecoderHFConfig.register_for_auto_class()
PicoDecoderHF.register_for_auto_class("AutoModel")
PicoDecoderHF.register_for_auto_class("AutoModelForCausalLM")
''')
src = open(PICO_PATH).read()
assert "self.embedding_proj    = nn.Embedding" in src
assert "self.pico_decoder =" not in src
# assert "return {}" in src
assert "if not attention_n_kv_heads:" in src
print(f"✅ pico_decoder.py written ({os.path.getsize(PICO_PATH):,} bytes)")

# =========================================================
# CONFIG TEMPLATE
# =========================================================
BASE_CONFIG = {
    "architectures": ["PicoDecoderHF"],
    "model_type": "pico_decoder",
    "auto_map": {
        "AutoConfig": "pico_decoder.PicoDecoderHFConfig",
        "AutoModelForCausalLM": "pico_decoder.PicoDecoderHF",
    },
    "n_layers": 14,
    "d_model": 768,
    "attention_n_heads": 12,
    "attention_n_kv_heads": 1,
    "max_seq_len": 512,
    "batch_size": 64,
    "position_emb_theta": 10000.0,
    "activation_hidden_dim": 3072,
    "norm_eps": 1e-5,
    "dropout": 0.1,
    "torch_dtype": "float32",
}

# =========================================================
# GET VOCAB SIZE
# =========================================================
def get_vocab(repo):
    branches = [b.name for b in api.list_repo_refs(repo).branches]
    steps = sorted([b for b in branches if re.match(r"step-\d+", b)],
                   key=lambda x: int(x.split("-")[1]))

    for b in steps:
        try:
            path = hf_hub_download(repo, "tokenizer.json", revision=b)
            data = json.load(open(path))
            vocab = data.get("model", {}).get("vocab", {})
            if vocab:
                return len(vocab)
        except:
            continue
    return None

# =========================================================
# PUSH FUNCTION (with retry + rate limit handling)
# =========================================================
def push_repo(repo):
    print(f"\n🚀 {repo}")

    vocab = get_vocab(repo)
    if not vocab:
        print("  ❌ no vocab"); return

    config = {**BASE_CONFIG, "vocab_size": vocab}

    branches = [b.name for b in api.list_repo_refs(repo).branches]
    steps = sorted([b for b in branches if re.match(r"step-\d+", b)],
                   key=lambda x: int(x.split("-")[1]))

    if "main" in branches:
        steps.append("main")

    for b in steps:
        success = False
        while not success:
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    cp = os.path.join(tmp, "config.json")
                    with open(cp, "w") as f:
                        json.dump(config, f, indent=2)

                    api.create_commit(
                        repo_id=repo,
                        repo_type="model",
                        operations=[
                            CommitOperationAdd("pico_decoder.py", PICO_PATH),
                            CommitOperationAdd("config.json", cp),
                        ],
                        commit_message=f"Fix HF compatibility (vocab={vocab})",
                        revision=b,
                    )

                print(f"  ✅ {b}")
                success = True

                time.sleep(1)

            except Exception as e:
                msg = str(e)

                if "429" in msg or "rate" in msg.lower():
                    print(f"  ⏳ rate limit → sleeping 60s")
                    time.sleep(60)
                elif "No files have been modified" in msg:
                    print(f"  ⏭ {b} (unchanged)")
                    success = True
                else:
                    print(f"  ❌ {b}: {msg[:80]}")
                    success = True  # skip hard failures

# =========================================================
# RUN ALL
# =========================================================
for repo in ALL_REPOS:
    push_repo(repo)

print("\n🎉 DONE — all models processed")
