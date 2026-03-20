"""
utils.py — Shared utilities for the BeetleLM analysis pipeline.

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

# FLORES-200 HuggingFace dataset id and split to use for PPL evaluation.
FLORES_DATASET = "facebook/flores"
FLORES_SPLIT   = "devtest"          # 1012 sentences per language

# Goldfish: only score the first GOLDFISH_TOKENS tokens of each sentence.
# Set to None to disable (score full sentence).
GOLDFISH_TOKENS: int | None = 512

# Map BeetleLM ISO codes → FLORES language tags
# FLORES uses full BCP-47-ish tags; extend as needed.
FLORES_LANG_MAP: dict[str, str] = {
    "nld": "nld_Latn",
    "deu": "deu_Latn",
    "zho": "zho_Hans",
    "fra": "fra_Latn",
    "fas": "pes_Arab",  # Western Persian
    "bul": "bul_Cyrl",
    "eng": "eng_Latn",
    "ukr": "ukr_Cyrl",
    "ind": "ind_Latn",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def load_model_and_tokenizer(
    repo: str,
    device: str = DEVICE,
) -> tuple:
    """Load (model, tokenizer) from HuggingFace hub. LRU-cached by repo."""
    print(f"[utils] Loading {repo} …")
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.eval()
    model.to(device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# FLORES sentence loading
# ---------------------------------------------------------------------------

def load_flores_sentences(iso_code: str) -> list[str]:
    """
    Return the list of FLORES devtest sentences for *iso_code*.

    iso_code should be a BeetleLM 3-letter code (e.g. "nld", "deu").
    """
    flores_tag = FLORES_LANG_MAP.get(iso_code)
    if flores_tag is None:
        raise ValueError(
            f"No FLORES tag found for ISO code '{iso_code}'. "
            f"Add it to FLORES_LANG_MAP in utils.py."
        )
    dataset = load_dataset(FLORES_DATASET, flores_tag, split=FLORES_SPLIT, trust_remote_code=True)
    return [example["sentence"] for example in dataset]


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
) -> list[float]:
    """
    Score a list of sentences under *repo*.

    Returns a list of per-sentence mean NLL values (same order as input).
    """
    model, tokenizer = load_model_and_tokenizer(repo)
    results: list[float] = []
    for i, sent in enumerate(sentences):
        nll = sentence_log_likelihood(sent, model, tokenizer, goldfish_tokens)
        results.append(nll)
        if verbose and (i + 1) % 100 == 0:
            print(f"  [{repo}] scored {i+1}/{len(sentences)}")
    return results
