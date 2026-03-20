"""
convergence.py — Checkpoint-level convergence and embedding drift analysis.

Mirrors the checkpoint infrastructure from eval_model.py:
  - Uses `list_checkpoints()` + the step-N branch naming convention
  - Loads one checkpoint at a time, frees it before loading the next
    (same RAM contract as eval_model.py)

Three tracking signals across training steps
--------------------------------------------
1. PPL TRAJECTORY
   FLORES mean PPL at every step-N checkpoint → convergence curves.
   For bilingual models: track BOTH L1 and L2 PPL simultaneously to see
   when/if the L1-forgetting inflection point occurs.

2. EMBEDDING DRIFT (per checkpoint)
   For each checkpoint, compute cosine distance of probe-word embeddings
   relative to the STEP-0 / main checkpoint of the SAME model.
   This shows how fast and how far representations move during training,
   rather than comparing to the mono baseline.

3. REPRESENTATIONAL SIMILARITY ACROSS STEPS (CKA trajectory)
   CKA between consecutive checkpoints — measures how much the representation
   changes per training step. A flat CKA ≈ 1 means training has converged;
   a drop signals a phase transition.

Outputs
-------
results/convergence/
  ppl_traj_<repo_slug>.csv       — step, lang, mean_ppl, mean_nll
  drift_traj_<repo_slug>.csv     — step, word, cosine_dist_from_step0
  cka_traj_<repo_slug>.csv       — step_a, step_b, cka  (consecutive pairs)
  all_ppl_trajectories.csv       — concatenated PPL trajectories, all models
"""

from __future__ import annotations

import gc
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from models import MODEL_GROUPS, get_bilingual_type, get_lang_pair
from utils import (
    FLORES_LANG_MAP,
    GOLDFISH_TOKENS,
    load_flores_sentences,
    nll_to_ppl,
    sentence_log_likelihood,
)
from embedding_drift import PROBE_WORDS, token_ids_for_words

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR   = Path("results/convergence")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# How many FLORES sentences to use for the PPL trajectory.
# Full devtest (1012) is accurate; 200 is fast for iteration.
PPL_SAMPLE_SIZE: int = 200


# ---------------------------------------------------------------------------
# Checkpoint listing (mirrors eval_model.py — no dependency on it)
# ---------------------------------------------------------------------------

def _step_number(name: str) -> Optional[int]:
    m = re.match(r"^step-(\d+)$", name)
    return int(m.group(1)) if m else None


def list_checkpoints(repo: str, hf_token: Optional[str] = None) -> list[str]:
    """
    Return all branch/revision names for *repo* that match 'step-N',
    sorted by step number (ascending). Includes 'main' as step 0.

    Uses the HuggingFace Hub API — no model weights are downloaded.
    """
    try:
        from huggingface_hub import list_repo_refs
        refs = list_repo_refs(repo, token=hf_token)
        step_branches = sorted(
            [r.name for r in refs.branches if _step_number(r.name) is not None],
            key=lambda n: _step_number(n),
        )
        # Prepend 'main' as the step-0 anchor
        return ["main"] + step_branches
    except Exception as e:
        logger.warning(f"[list_checkpoints] {repo}: {e} — returning ['main'] only")
        return ["main"]


def step_label(revision: str) -> int:
    """Convert 'main' → 0, 'step-N' → N."""
    n = _step_number(revision)
    return n if n is not None else 0


# ---------------------------------------------------------------------------
# Checkpoint loading (same RAM contract as eval_model.py)
# ---------------------------------------------------------------------------

def _load_checkpoint(
    repo: str,
    revision: str,
    hf_token: Optional[str] = None,
    device: str = DEVICE,
) -> tuple:
    """Load (model, tokenizer) for one checkpoint. Caller must del both."""
    load_kw = dict(
        revision          = revision,
        trust_remote_code = True,
        torch_dtype       = torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage = True,
    )
    if hf_token:
        load_kw["token"] = hf_token

    model = (
        AutoModelForCausalLM
        .from_pretrained(repo, **load_kw)
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(repo, **load_kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _free(model, tokenizer) -> None:
    """Deterministic GPU/CPU memory release — mirrors eval_model.py."""
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Signal 1: PPL trajectory
# ---------------------------------------------------------------------------

def _ppl_for_checkpoint(
    model,
    tokenizer,
    sentences: list[str],
    goldfish_tokens: int | None = GOLDFISH_TOKENS,
) -> float:
    """Mean NLL across sentences for a loaded model."""
    nlls = [
        sentence_log_likelihood(s, model, tokenizer, goldfish_tokens)
        for s in sentences
    ]
    return float(np.mean([n for n in nlls if n < float("inf")]))


def compute_ppl_trajectory(
    repo: str,
    lang_codes: list[str],
    hf_token: Optional[str] = None,
    sample_size: int = PPL_SAMPLE_SIZE,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    For each checkpoint of *repo*, compute mean PPL for each language in
    *lang_codes*.

    lang_codes should include BOTH the L1 and L2 languages of the model so
    that forgetting inflection points are visible.

    Returns DataFrame: [step, revision, lang, mean_nll, mean_ppl]
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug     = repo.replace("/", "__")
    out_path = OUTPUT_DIR / f"ppl_traj_{slug}.csv"

    if out_path.exists() and not overwrite:
        logger.info(f"[ppl_traj] Loading cached {out_path}")
        return pd.read_csv(out_path)

    # Pre-load sentences (outside the checkpoint loop)
    sentences_by_lang: dict[str, list[str]] = {}
    for code in lang_codes:
        if code not in FLORES_LANG_MAP:
            logger.warning(f"[ppl_traj] No FLORES mapping for {code!r} — skipping")
            continue
        all_sents = load_flores_sentences(code)
        # Deterministic subsample
        step = max(1, len(all_sents) // sample_size)
        sentences_by_lang[code] = all_sents[::step][:sample_size]
        logger.info(f"[ppl_traj] {code}: using {len(sentences_by_lang[code])} sentences")

    checkpoints = list_checkpoints(repo, hf_token)
    logger.info(f"[ppl_traj] {repo}: {len(checkpoints)} checkpoints × {len(sentences_by_lang)} langs")

    rows: list[dict] = []

    for revision in checkpoints:
        step = step_label(revision)
        logger.info(f"  step {step:>6d} ({revision}) …")

        try:
            model, tokenizer = _load_checkpoint(repo, revision, hf_token)
        except Exception as e:
            logger.error(f"  Load failed @ {revision}: {e}")
            continue

        for code, sentences in sentences_by_lang.items():
            mean_nll = _ppl_for_checkpoint(model, tokenizer, sentences)
            rows.append({
                "repo":      repo,
                "revision":  revision,
                "step":      step,
                "lang":      code,
                "mean_nll":  round(mean_nll, 4),
                "mean_ppl":  round(nll_to_ppl(mean_nll), 2),
            })
            logger.info(f"    {code}: PPL={nll_to_ppl(mean_nll):.1f}")

        _free(model, tokenizer)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info(f"[ppl_traj] saved {out_path}")
    return df


# ---------------------------------------------------------------------------
# Signal 2: Probe-word embedding drift across checkpoints
# ---------------------------------------------------------------------------

def _probe_vecs_for_checkpoint(
    model,
    tokenizer,
    probe_words: list[str],
) -> dict[str, np.ndarray]:
    """
    Extract embedding vector for each probe word (single-token only).
    Returns {word: vector [hidden_dim]} for words that tokenise to 1 token.
    """
    emb_weight = model.get_input_embeddings().weight.detach().float().cpu()
    probe_map  = token_ids_for_words(probe_words, tokenizer)
    result: dict[str, np.ndarray] = {}
    for word, ids in probe_map.items():
        if len(ids) == 1:
            result[word] = emb_weight[ids[0]].numpy()
    return result


def compute_drift_trajectory(
    repo: str,
    iso_code: str,
    hf_token: Optional[str] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    For each checkpoint, compute cosine distance of each probe word's embedding
    relative to its embedding at step 0 (the 'main' / earliest checkpoint).

    This gives you the per-word drift curve across training.

    Returns DataFrame: [step, revision, word, cosine_dist_from_step0,
                        cosine_dist_from_prev]
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug     = repo.replace("/", "__")
    out_path = OUTPUT_DIR / f"drift_traj_{slug}.csv"

    if out_path.exists() and not overwrite:
        logger.info(f"[drift_traj] Loading cached {out_path}")
        return pd.read_csv(out_path)

    probe_words  = PROBE_WORDS.get(iso_code, [])
    checkpoints  = list_checkpoints(repo, hf_token)
    logger.info(f"[drift_traj] {repo}: {len(checkpoints)} checkpoints, {len(probe_words)} probe words")

    # Step-0 anchor: extract embeddings at 'main'
    logger.info(f"  Loading step-0 anchor (main) …")
    try:
        model0, tok0 = _load_checkpoint(repo, "main", hf_token)
        step0_vecs   = _probe_vecs_for_checkpoint(model0, tok0, probe_words)
        _free(model0, tok0)
    except Exception as e:
        logger.error(f"  Could not load step-0 anchor: {e}")
        return pd.DataFrame()

    rows: list[dict] = []
    prev_vecs: dict[str, np.ndarray] = step0_vecs.copy()

    for revision in checkpoints:
        step = step_label(revision)
        logger.info(f"  step {step:>6d} ({revision}) …")

        try:
            model, tokenizer = _load_checkpoint(repo, revision, hf_token)
            ckpt_vecs = _probe_vecs_for_checkpoint(model, tokenizer, probe_words)
            _free(model, tokenizer)
        except Exception as e:
            logger.error(f"  Load failed @ {revision}: {e}")
            continue

        for word in step0_vecs:
            if word not in ckpt_vecs:
                continue

            v0   = torch.tensor(step0_vecs[word]).unsqueeze(0)
            vt   = torch.tensor(ckpt_vecs[word]).unsqueeze(0)
            vprev= torch.tensor(prev_vecs.get(word, step0_vecs[word])).unsqueeze(0)

            dist_from_step0 = 1.0 - cosine_similarity(v0, vt).item()
            dist_from_prev  = 1.0 - cosine_similarity(vprev, vt).item()

            rows.append({
                "repo":                repo,
                "iso_code":            iso_code,
                "revision":            revision,
                "step":                step,
                "word":                word,
                "cosine_dist_step0":   round(dist_from_step0, 5),
                "cosine_dist_prev":    round(dist_from_prev, 5),
            })

        prev_vecs = ckpt_vecs

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info(f"[drift_traj] saved {out_path}")
    return df


# ---------------------------------------------------------------------------
# Signal 3: Consecutive-checkpoint CKA
# ---------------------------------------------------------------------------

def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two representation matrices [n_probes, hidden_dim]."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    K_X = X @ X.T
    K_Y = Y @ Y.T
    n   = K_X.shape[0]
    H   = np.eye(n) - np.ones((n, n)) / n
    cK_X = H @ K_X @ H
    cK_Y = H @ K_Y @ H
    num  = np.sum(cK_X * cK_Y)
    den  = (np.linalg.norm(cK_X, "fro") * np.linalg.norm(cK_Y, "fro")) + 1e-10
    return float(num / den)


def compute_cka_trajectory(
    repo: str,
    iso_code: str,
    hf_token: Optional[str] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Compute CKA between every pair of consecutive checkpoints.

    A CKA close to 1 between step-N and step-N+k means representations are
    stable (converged). A sudden drop indicates a phase transition.

    Returns DataFrame: [step_a, step_b, revision_a, revision_b, cka,
                        delta_step]
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug     = repo.replace("/", "__")
    out_path = OUTPUT_DIR / f"cka_traj_{slug}.csv"

    if out_path.exists() and not overwrite:
        logger.info(f"[cka_traj] Loading cached {out_path}")
        return pd.read_csv(out_path)

    probe_words = PROBE_WORDS.get(iso_code, [])
    checkpoints = list_checkpoints(repo, hf_token)

    # Collect probe vectors checkpoint-by-checkpoint.
    # Only keep ONE checkpoint's vectors in memory at a time.
    rows: list[dict] = []

    prev_vecs: dict[str, np.ndarray] | None = None
    prev_step  = None
    prev_rev   = None

    for revision in checkpoints:
        step = step_label(revision)
        logger.info(f"  step {step:>6d} ({revision}) …")

        try:
            model, tokenizer = _load_checkpoint(repo, revision, hf_token)
            ckpt_vecs        = _probe_vecs_for_checkpoint(model, tokenizer, probe_words)
            _free(model, tokenizer)
        except Exception as e:
            logger.error(f"  Load failed @ {revision}: {e}")
            continue

        if prev_vecs is not None:
            # Compute CKA over shared probe words only
            shared = sorted(set(prev_vecs.keys()) & set(ckpt_vecs.keys()))
            if len(shared) >= 2:
                X = np.stack([prev_vecs[w]  for w in shared])
                Y = np.stack([ckpt_vecs[w] for w in shared])
                cka = _linear_cka(X, Y)
                rows.append({
                    "repo":       repo,
                    "iso_code":   iso_code,
                    "revision_a": prev_rev,
                    "revision_b": revision,
                    "step_a":     prev_step,
                    "step_b":     step,
                    "delta_step": step - prev_step,
                    "cka":        round(cka, 5),
                    # 1 - CKA = representational change per unit step
                    "change":     round(1.0 - cka, 5),
                })
                logger.info(f"    CKA({prev_step}→{step}) = {cka:.4f}")

        prev_vecs = ckpt_vecs
        prev_step = step
        prev_rev  = revision

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info(f"[cka_traj] saved {out_path}")
    return df


# ---------------------------------------------------------------------------
# Batch runner: all three signals for a list of repos
# ---------------------------------------------------------------------------

def _iso_codes_for_repo(repo: str) -> list[str]:
    """Return the ISO code(s) a repo belongs to (from MODEL_GROUPS)."""
    return [code for code, models in MODEL_GROUPS.items() if repo in models]


def _lang_pair_codes(repo: str) -> list[str]:
    """
    Parse L1 and L2 ISO codes from the repo name so we can track both
    in the PPL trajectory.

    e.g. beetlelm_eng-nld_balanced → ['eng', 'nld']
         beetlelm_nld_mono         → ['nld']
    """
    name  = repo.split("/")[-1].replace("beetlelm_", "")
    for tag in ("_mono", "_balanced", "_simultaneous", "_sequential",
                "_part_time", "_late", "_heritage"):
        name = name.replace(tag, "")
    # name is now e.g. "eng-nld" or "nld_L1-eng_L2" or "nld"
    # Normalise L1/L2 labels and extract raw iso codes
    name = re.sub(r"_L[12]", "", name)          # strip _L1 _L2 suffixes
    codes = re.split(r"[-_]", name)
    # Keep only recognised 3-letter codes
    known = set(FLORES_LANG_MAP.keys())
    return [c for c in codes if c in known]


def run_convergence_analysis(
    repos: list[str],
    hf_token: Optional[str] = None,
    overwrite: bool = False,
    signals: list[str] = ("ppl", "drift", "cka"),
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Run all convergence signals for a list of repos.

    Args:
        repos:    HuggingFace repo strings (from models.py).
        signals:  Subset of ('ppl', 'drift', 'cka') to compute.
        overwrite: Re-compute even if CSVs exist.

    Returns:
        { repo: { 'ppl': df, 'drift': df, 'cka': df } }
    """
    results: dict[str, dict] = {}

    for repo in repos:
        iso_codes = _iso_codes_for_repo(repo)
        lang_pair = _lang_pair_codes(repo)
        primary_iso = iso_codes[0] if iso_codes else (lang_pair[0] if lang_pair else None)

        logger.info(f"\n{'='*60}")
        logger.info(f"Convergence: {repo}")
        logger.info(f"  ISO groups : {iso_codes}")
        logger.info(f"  PPL langs  : {lang_pair}")
        logger.info(f"  Primary iso: {primary_iso}")

        results[repo] = {}

        if "ppl" in signals and lang_pair:
            results[repo]["ppl"] = compute_ppl_trajectory(
                repo, lang_pair, hf_token=hf_token, overwrite=overwrite
            )

        if "drift" in signals and primary_iso:
            results[repo]["drift"] = compute_drift_trajectory(
                repo, primary_iso, hf_token=hf_token, overwrite=overwrite
            )

        if "cka" in signals and primary_iso:
            results[repo]["cka"] = compute_cka_trajectory(
                repo, primary_iso, hf_token=hf_token, overwrite=overwrite
            )

    # Concatenate all PPL trajectories into a single overview CSV
    ppl_frames = [v["ppl"] for v in results.values() if "ppl" in v and not v["ppl"].empty]
    if ppl_frames:
        combined = pd.concat(ppl_frames, ignore_index=True)
        combined_path = OUTPUT_DIR / "all_ppl_trajectories.csv"
        combined.to_csv(combined_path, index=False)
        logger.info(f"\n[convergence] All PPL trajectories → {combined_path}")

    return results


# ---------------------------------------------------------------------------
# Convenience: run for the key comparison models from the research questions
# ---------------------------------------------------------------------------

FOCUS_REPOS = {
    # L1 forgetting focus (deu, nld, zho)
    "forgetting_focus": [
        "BeetleLM/beetlelm_nld_mono",
        "BeetleLM/beetlelm_eng-nld_simultaneous",
        "BeetleLM/beetlelm_eng-nld_balanced",
        "BeetleLM/beetlelm_eng-nld_late",
        "BeetleLM/beetlelm_deu_mono",
        "BeetleLM/beetlelm_eng-deu_simultaneous",
        "BeetleLM/beetlelm_bul-deu_sequential",
        "BeetleLM/beetlelm_zho_mono",
        "BeetleLM/beetlelm_zho-eng_balanced",
        "BeetleLM/beetlelm_zho_L1-eng_L2_simultaneous",
    ],
    # Reading time / L2 focus
    "rt_focus": [
        "BeetleLM/beetlelm_eng-nld_simultaneous",      # simultaneous → L2 RT
        "BeetleLM/beetlelm_nld_L1-eng_L2_simultaneous",
        "BeetleLM/beetlelm_zho-eng_balanced",           # Chinese/English balanced
        "BeetleLM/beetlelm_zho_L1-eng_L2_balanced",
    ],
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run checkpoint convergence analysis.")
    parser.add_argument("--repos",     nargs="+",
                        help="Explicit repo list (default: forgetting_focus set)")
    parser.add_argument("--focus",     choices=list(FOCUS_REPOS.keys()),
                        default="forgetting_focus")
    parser.add_argument("--signals",   nargs="+",
                        choices=["ppl", "drift", "cka"],
                        default=["ppl", "drift", "cka"])
    parser.add_argument("--hf_token",  default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repos = args.repos if args.repos else FOCUS_REPOS[args.focus]
    run_convergence_analysis(
        repos,
        hf_token  = args.hf_token,
        overwrite = args.overwrite,
        signals   = args.signals,
    )
