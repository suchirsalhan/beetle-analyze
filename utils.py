"""
utils.py — Shared utilities for BeetleLM evaluation:
  - Model loading / cleanup
  - Batched log-probability scoring
  - Checkpoint enumeration from HF Hub
  - Thread-safe CSV appending
"""

import re
import os
import csv
import fcntl
import logging
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

# ── HF API (singleton) ────────────────────────────────────────────────────────
_api: Optional[HfApi] = None

def get_api(token: Optional[str] = None) -> HfApi:
    global _api
    if _api is None:
        _api = HfApi(token=token)
    return _api


# ── Checkpoint enumeration ────────────────────────────────────────────────────

def list_checkpoints(repo: str, token: Optional[str] = None) -> List[str]:
    """
    Return ordered list of branches to evaluate:
      step-100, step-200, ..., main
    Falls back to ['main'] if branch listing fails.
    """
    api = get_api(token)
    try:
        refs  = api.list_repo_refs(repo_id=repo, token=token)
        names = [b.name for b in refs.branches]
        steps = sorted(
            [n for n in names if re.match(r"^step-\d+$", n)],
            key=lambda n: int(n.split("-")[1])
        )
        ordered = steps + (["main"] if "main" in names else [])
        return ordered if ordered else ["main"]
    except Exception as e:
        logger.warning(f"Could not list branches for {repo}: {e}. Using 'main'.")
        return ["main"]


# ── Model loading / cleanup ───────────────────────────────────────────────────

def load_model_and_tokenizer(
    repo: str,
    branch: str,
    device: torch.device,
    token: Optional[str] = None,
):
    """Load a causal LM and its tokenizer onto `device`."""
    kwargs = dict(
        revision         = branch,
        trust_remote_code = True,
        torch_dtype      = torch.float16,   # bf16 on A100 is also fine
        token            = token,
    )
    model = (
        AutoModelForCausalLM
        .from_pretrained(repo, **kwargs)
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(repo, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def release(model, tokenizer, device: torch.device):
    """Delete model + tokenizer and free GPU memory."""
    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ── Scoring ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_sentences(
    model,
    tokenizer,
    sentences: List[str],
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Returns a 1-D tensor of total log-probabilities for each sentence.
    Handles arbitrary-length lists via internal batching.
    """
    all_scores = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        enc   = tokenizer(
            batch,
            padding        = True,
            truncation     = True,
            max_length     = 512,
            return_tensors = "pt",
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # clamp to valid vocab range (safety)
        input_ids = input_ids.clamp(0, model.config.vocab_size - 1)

        outputs   = model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = F.log_softmax(outputs.logits, dim=-1)          # [B, T, V]

        # shift: predict token t from token t-1
        lp_shifted  = log_probs[:, :-1, :]                         # [B, T-1, V]
        ids_shifted = input_ids[:, 1:]                             # [B, T-1]
        mask_shifted= attention_mask[:, 1:].float()                # [B, T-1]

        token_lp = lp_shifted.gather(-1, ids_shifted.unsqueeze(-1)).squeeze(-1)
        # sum only over real (non-padding) tokens
        sentence_lp = (token_lp * mask_shifted).sum(dim=-1)       # [B]
        all_scores.append(sentence_lp.cpu())

    return torch.cat(all_scores)


def minimal_pair_accuracy(
    model,
    tokenizer,
    pairs: List[Tuple[str, str]],
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[float, int, int]:
    """
    Evaluate minimal-pair accuracy.
    Returns (accuracy, n_correct, n_total).
    """
    good_sents = [p[0] for p in pairs]
    bad_sents  = [p[1] for p in pairs]

    scores_good = score_sentences(model, tokenizer, good_sents, device, batch_size)
    scores_bad  = score_sentences(model, tokenizer, bad_sents,  device, batch_size)

    n_correct = int((scores_good > scores_bad).sum().item())
    n_total   = len(pairs)
    accuracy  = n_correct / n_total if n_total > 0 else 0.0
    return accuracy, n_correct, n_total


# ── Thread-safe CSV writing ───────────────────────────────────────────────────

RESULT_FIELDS = [
    "benchmark", "model", "lang_pair", "bilingual_type",
    "checkpoint", "eval_language", "accuracy", "n_correct", "n_total",
]

def append_result(csv_path: str, row: dict):
    """
    Append a single result row to a CSV file.
    Uses flock so multiple GPU processes can write to the same file safely.
    """
    path   = Path(csv_path)
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        fcntl.flock(f, fcntl.LOCK_UN)


def already_done(csv_path: str, model: str, checkpoint: str, eval_language: str,
                 benchmark: str) -> bool:
    """
    Check if a (model, checkpoint, eval_language, benchmark) combination
    is already in the CSV — lets you resume interrupted runs without
    re-evaluating.
    """
    path = Path(csv_path)
    if not path.exists():
        return False
    import pandas as pd
    df = pd.read_csv(path)
    mask = (
        (df["model"]         == model)       &
        (df["checkpoint"]    == checkpoint)  &
        (df["eval_language"] == eval_language) &
        (df["benchmark"]     == benchmark)
    )
    return bool(mask.any())
