"""
ppl_utils.py — Shared utilities for the BeetleLM analysis pipeline.
(Named ppl_utils to avoid collision with the repo-root utils.py used by eval_model.py.)

Provides:
  - Model / tokenizer loading with caching
  - Sentence-level log-likelihood (token-mean NLL → PPL)
  - FLORES-200 sentence loader per ISO-639-3 language code
  - Goldfish truncation (first-N-token filtering)
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Iterator

import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FLORES_SPLIT = "devtest"   # 1012 sentences per language

# Goldfish: only score the first GOLDFISH_TOKENS tokens of each sentence.
# Set to None to disable (score full sentence).
GOLDFISH_TOKENS: int | None = 512

# Map BeetleLM ISO codes → FLORES-200 BCP-47 language tags
FLORES_LANG_MAP: dict[str, str] = {
    "nld": "nld_Latn",
    "deu": "deu_Latn",
    "zho": "zho_Hans",
    "fra": "fra_Latn",
    "fas": "pes_Arab",
    "bul": "bul_Cyrl",
    "eng": "eng_Latn",
    "ukr": "ukr_Cyrl",
    "ind": "ind_Latn",
}


# ---------------------------------------------------------------------------
# FLORES sentence loading — direct parquet, no dataset scripts
# ---------------------------------------------------------------------------

def load_flores_sentences(iso_code: str) -> list[str]:
    """
    Return FLORES-200 devtest sentences for *iso_code*.

    Uses crystina-z/flores200 — a parquet-native dataset with all 200
    languages as columns named sentence_{tag} (e.g. sentence_nld_Latn).
    No legacy dataset script; loads cleanly with any datasets version.
    """
    flores_tag = FLORES_LANG_MAP.get(iso_code)
    if flores_tag is None:
        raise ValueError(
            f"No FLORES tag for '{iso_code}'. Add to FLORES_LANG_MAP in ppl_utils.py."
        )

    col = f"sentence_{flores_tag}"
    ds  = load_dataset("crystina-z/flores200", "all", split="devtest",
                       trust_remote_code=False)

    if col not in ds.column_names:
        raise ValueError(
            f"Column '{col}' not in crystina-z/flores200. "
            f"Available: {[c for c in ds.column_names if c.startswith('sentence_')][:10]}"
        )

    sentences = [s for s in ds[col] if isinstance(s, str) and s.strip()]
    print(f"[flores] {iso_code}: {len(sentences)} sentences (col={col})")
    return sentences



# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def get_best_revision(repo: str, hf_token: str | None = None) -> str:
    """
    Return the highest step-N branch name for *repo*.
    Falls back to 'main' if no step-N branches exist.
    """
    import re
    try:
        from huggingface_hub import list_repo_refs
        refs = list_repo_refs(repo, token=hf_token)
        step_branches = [
            (int(m.group(1)), r.name)
            for r in refs.branches
            if (m := re.match(r"^step-(\d+)$", r.name))
        ]
        if step_branches:
            best = max(step_branches, key=lambda t: t[0])[1]
            print(f"[utils] {repo}: using checkpoint {best}")
            return best
    except Exception as e:
        print(f"[utils] Could not list refs for {repo}: {e}")
    print(f"[utils] {repo}: no step-N branches found, falling back to main")
    return "main"


# Registry of repos that failed to load — populated by load_model_and_tokenizer,
# read by ppl_eval / forgetting / embedding_drift to print a skip summary.
LOAD_FAILURES: dict[str, str] = {}


@lru_cache(maxsize=4)
def load_model_and_tokenizer(
    repo: str,
    device: str = DEVICE,
) -> tuple | None:
    """
    Load (model, tokenizer) from the highest step-N checkpoint. LRU-cached.

    Returns None (and records the error in LOAD_FAILURES) if the model
    cannot be loaded, so callers can skip gracefully rather than crashing.
    """
    revision = get_best_revision(repo)
    print(f"[utils] Loading {repo} @ {revision} …")
    load_kw = dict(revision=revision, trust_remote_code=True)
    try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo, **load_kw, use_fast=True)
        except (ValueError, ImportError):
            print(f"[utils] Fast tokenizer unavailable for {repo}, using slow tokenizer.")
            tokenizer = AutoTokenizer.from_pretrained(repo, **load_kw, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            **load_kw,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )
        model.eval()
        model.to(device)
        return model, tokenizer
    except Exception as exc:
        msg = str(exc).split("\n")[0]          # first line only — keep logs tidy
        print(f"[utils] SKIP {repo}: {msg}")
        LOAD_FAILURES[repo] = msg
        load_model_and_tokenizer.cache_clear()  # don't cache the None result
        return None


# ---------------------------------------------------------------------------
# Log-likelihood / perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def sentence_log_likelihood(
    sentence: str,
    model,
    tokenizer,
    goldfish_tokens: int | None = GOLDFISH_TOKENS,
    stride: int = 512,
) -> float:
    """
    Return the mean-token negative log-likelihood (NLL) for *sentence*.

    Lower is better (more likely under the model).
    Perplexity = exp(NLL).

    Args:
        sentence:        The raw string to score.
        model:           A loaded CausalLM.
        tokenizer:       Matching tokenizer.
        goldfish_tokens: If set, only score the first N tokens (Goldfish
                         evaluation protocol). Set to None for full sentence.
        stride:          Sliding-window stride for long sequences.

    Returns:
        Mean NLL (float). Returns float("inf") on tokenisation error.
    """
    encodings = tokenizer(sentence, return_tensors="pt")
    input_ids: torch.Tensor = encodings.input_ids.to(model.device)
    seq_len = input_ids.size(1)

    # Goldfish truncation
    if goldfish_tokens is not None and seq_len > goldfish_tokens:
        input_ids = input_ids[:, :goldfish_tokens]
        seq_len = goldfish_tokens

    if seq_len < 2:
        return float("inf")

    loss_fn = CrossEntropyLoss(reduction="sum")
    nlls: list[float] = []
    token_count = 0

    max_len = getattr(model.config, "max_position_embeddings", 2048)
    window = min(max_len, 1024)

    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + window, seq_len)
        target_len = end - prev_end

        chunk = input_ids[:, begin:end]
        labels = chunk.clone()
        # Mask out the context tokens that were already scored
        labels[:, :-target_len] = -100

        with torch.amp.autocast("cuda", enabled=(model.device.type == "cuda")):
            outputs = model(chunk, labels=labels)

        # outputs.loss is mean NLL over non-masked tokens; recover sum
        num_scored = (labels != -100).sum().item() - 1  # shift by 1
        if num_scored <= 0:
            prev_end = end
            continue

        nlls.append(outputs.loss.item() * num_scored)
        token_count += num_scored
        prev_end = end

        if end == seq_len:
            break

    if token_count == 0:
        return float("inf")

    return sum(nlls) / token_count  # mean NLL across all scored tokens


def nll_to_ppl(nll: float) -> float:
    """Convert mean NLL to perplexity."""
    return math.exp(nll) if nll < 700 else float("inf")


# ---------------------------------------------------------------------------
# Batch scoring helper
# ---------------------------------------------------------------------------

def score_sentences(
    sentences: list[str],
    repo: str,
    goldfish_tokens: int | None = GOLDFISH_TOKENS,
    verbose: bool = True,
) -> list[float] | None:
    """
    Score a list of sentences under *repo*.

    Returns a list of per-sentence mean NLL values, or None if the model
    failed to load (caller should skip this repo).
    """
    result = load_model_and_tokenizer(repo)
    if result is None:
        return None                              # load failed — caller skips
    model, tokenizer = result
    nlls: list[float] = []
    for i, sent in enumerate(sentences):
        nll = sentence_log_likelihood(sent, model, tokenizer, goldfish_tokens)
        nlls.append(nll)
        if verbose and (i + 1) % 100 == 0:
            print(f"  [{repo}] scored {i+1}/{len(sentences)}")
    return nlls