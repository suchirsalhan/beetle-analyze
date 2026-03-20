"""
convergence.py — Checkpoint-level convergence and embedding drift analysis.

Mirrors the checkpoint infrastructure from eval_model.py:
  - Uses `list_checkpoints()` + the step-N branch naming convention
  - Loads one checkpoint at a time, frees it before loading the next
    (same RAM contract as eval_model.py)

Three tracking signals across training steps
--------------------------------------------
1. PPL TRAJECTORY
2. EMBEDDING DRIFT (per checkpoint)
3. REPRESENTATIONAL SIMILARITY ACROSS STEPS (CKA trajectory)

Outputs
-------
results/convergence/
  ppl_traj_<repo_slug>.csv       — step, lang, mean_ppl, mean_nll
  drift_traj_<repo_slug>.csv     — step, word, cosine_dist_from_step0
  cka_traj_<repo_slug>.csv       — step_a, step_b, cka  (consecutive pairs)
  all_ppl_trajectories.csv       — concatenated PPL trajectories, all models
"""

from __future__ import annotations
import _path  # noqa: F401  — adds repo root to sys.path

import gc
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

from models import MODEL_GROUPS, get_bilingual_type, get_lang_pair
from ppl_utils import (
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

OUTPUT_DIR      = Path("results/convergence")
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
PPL_SAMPLE_SIZE = 200

# Registry of (repo, revision) pairs that failed to load.
CONVERGENCE_FAILURES: dict[str, str] = {}   # key = "repo@revision"


# ---------------------------------------------------------------------------
# Checkpoint listing
# ---------------------------------------------------------------------------

def _step_number(name: str) -> Optional[int]:
    m = re.match(r"^step-(\d+)$", name)
    return int(m.group(1)) if m else None


def list_checkpoints(repo: str, hf_token: Optional[str] = None) -> list[str]:
    try:
        from huggingface_hub import list_repo_refs
        refs = list_repo_refs(repo, token=hf_token)
        step_branches = sorted(
            [r.name for r in refs.branches if _step_number(r.name) is not None],
            key=lambda n: _step_number(n),
        )
        return ["main"] + step_branches
    except Exception as e:
        logger.warning(f"[list_checkpoints] {repo}: {e} — returning ['main'] only")
        return ["main"]


def step_label(revision: str) -> int:
    n = _step_number(revision)
    return n if n is not None else 0


# ---------------------------------------------------------------------------
# Checkpoint loading — returns None on failure, records in CONVERGENCE_FAILURES
# ---------------------------------------------------------------------------

def _load_checkpoint(
    repo: str,
    revision: str,
    hf_token: Optional[str] = None,
    device: str = DEVICE,
) -> tuple | None:
    """
    Load (model, tokenizer) for one checkpoint.
    Returns None and records the error if loading fails.
    Caller must del both when done.
    """
    load_kw = dict(
        revision          = revision,
        trust_remote_code = True,
        torch_dtype       = torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage = True,
    )
    if hf_token:
        load_kw["token"] = hf_token

    try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo, **load_kw, use_fast=True)
        except (ValueError, ImportError):
            tokenizer = AutoTokenizer.from_pretrained(repo, **load_kw, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = (
            AutoModelForCausalLM
            .from_pretrained(repo, **load_kw)
            .to(device)
            .eval()
        )
        return model, tokenizer

    except Exception as exc:
        msg = str(exc).split("\n")[0]
        key = f"{repo}@{revision}"
        logger.error(f"  SKIP {key}: {msg}")
        CONVERGENCE_FAILURES[key] = msg
        return None


def _free(model, tokenizer) -> None:
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Signal 1: PPL trajectory
# ---------------------------------------------------------------------------

def _ppl_for_checkpoint(model, tokenizer, sentences, goldfish_tokens=GOLDFISH_TOKENS):
    nlls = [sentence_log_likelihood(s, model, tokenizer, goldfish_tokens) for s in sentences]
    return float(np.mean([n for n in nlls if n < float("inf")]))


def compute_ppl_trajectory(
    repo: str,
    lang_codes: list[str],
    hf_token: Optional[str] = None,
    sample_size: int = PPL_SAMPLE_SIZE,
    overwrite: bool = False,
) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug     = repo.replace("/", "__")
    out_path = OUTPUT_DIR / f"ppl_traj_{slug}.csv"

    if out_path.exists() and not overwrite:
        logger.info(f"[ppl_traj] Loading cached {out_path}")
        return pd.read_csv(out_path)

    sentences_by_lang: dict[str, list[str]] = {}
    for code in lang_codes:
        if code not in FLORES_LANG_MAP:
            logger.warning(f"[ppl_traj] No FLORES mapping for {code!r} — skipping")
            continue
        all_sents = load_flores_sentences(code)
        step = max(1, len(all_sents) // sample_size)
        sentences_by_lang[code] = all_sents[::step][:sample_size]
        logger.info(f"[ppl_traj] {code}: {len(sentences_by_lang[code])} sentences")

    checkpoints = list_checkpoints(repo, hf_token)
    logger.info(f"[ppl_traj] {repo}: {len(checkpoints)} checkpoints × {len(sentences_by_lang)} langs")

    rows: list[dict] = []
    for revision in checkpoints:
        step   = step_label(revision)
        result = _load_checkpoint(repo, revision, hf_token)
        if result is None:
            logger.warning(f"  SKIPPED step {step} ({revision}) — load failed.")
            continue
        model, tokenizer = result

        for code, sentences in sentences_by_lang.items():
            mean_nll = _ppl_for_checkpoint(model, tokenizer, sentences)
            rows.append({
                "repo":     repo,
                "revision": revision,
                "step":     step,
                "lang":     code,
                "mean_nll": round(mean_nll, 4),
                "mean_ppl": round(nll_to_ppl(mean_nll), 2),
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

def _probe_vecs_for_checkpoint(model, tokenizer, probe_words):
    emb_weight = model.get_input_embeddings().weight.detach().float().cpu()
    probe_map  = token_ids_for_words(probe_words, tokenizer)
    return {word: emb_weight[ids[0]].numpy()
            for word, ids in probe_map.items() if len(ids) == 1}


def compute_drift_trajectory(
    repo: str,
    iso_code: str,
    hf_token: Optional[str] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug     = repo.replace("/", "__")
    out_path = OUTPUT_DIR / f"drift_traj_{slug}.csv"

    if out_path.exists() and not overwrite:
        logger.info(f"[drift_traj] Loading cached {out_path}")
        return pd.read_csv(out_path)

    probe_words = PROBE_WORDS.get(iso_code, [])
    checkpoints = list_checkpoints(repo, hf_token)
    logger.info(f"[drift_traj] {repo}: {len(checkpoints)} checkpoints, {len(probe_words)} probe words")

    logger.info("  Loading step-0 anchor (main) …")
    result0 = _load_checkpoint(repo, "main", hf_token)
    if result0 is None:
        logger.error(f"  SKIPPED {repo} drift — step-0 anchor failed to load.")
        return pd.DataFrame()
    model0, tok0 = result0
    step0_vecs   = _probe_vecs_for_checkpoint(model0, tok0, probe_words)
    _free(model0, tok0)

    rows: list[dict] = []
    prev_vecs = step0_vecs.copy()

    for revision in checkpoints:
        step   = step_label(revision)
        result = _load_checkpoint(repo, revision, hf_token)
        if result is None:
            logger.warning(f"  SKIPPED step {step} ({revision}) — load failed.")
            continue
        model, tokenizer = result
        ckpt_vecs = _probe_vecs_for_checkpoint(model, tokenizer, probe_words)
        _free(model, tokenizer)

        for word in step0_vecs:
            if word not in ckpt_vecs:
                continue
            v0    = torch.tensor(step0_vecs[word]).unsqueeze(0)
            vt    = torch.tensor(ckpt_vecs[word]).unsqueeze(0)
            vprev = torch.tensor(prev_vecs.get(word, step0_vecs[word])).unsqueeze(0)
            rows.append({
                "repo":              repo,
                "iso_code":          iso_code,
                "revision":          revision,
                "step":              step,
                "word":              word,
                "cosine_dist_step0": round(1.0 - cosine_similarity(v0, vt).item(), 5),
                "cosine_dist_prev":  round(1.0 - cosine_similarity(vprev, vt).item(), 5),
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
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    K_X = X @ X.T
    K_Y = Y @ Y.T
    n   = K_X.shape[0]
    H   = np.eye(n) - np.ones((n, n)) / n
    cKX = H @ K_X @ H
    cKY = H @ K_Y @ H
    return float(np.sum(cKX * cKY) / (np.linalg.norm(cKX, "fro") * np.linalg.norm(cKY, "fro") + 1e-10))


def compute_cka_trajectory(
    repo: str,
    iso_code: str,
    hf_token: Optional[str] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug     = repo.replace("/", "__")
    out_path = OUTPUT_DIR / f"cka_traj_{slug}.csv"

    if out_path.exists() and not overwrite:
        logger.info(f"[cka_traj] Loading cached {out_path}")
        return pd.read_csv(out_path)

    probe_words = PROBE_WORDS.get(iso_code, [])
    checkpoints = list_checkpoints(repo, hf_token)

    rows: list[dict] = []
    prev_vecs: dict[str, np.ndarray] | None = None
    prev_step = None
    prev_rev  = None

    for revision in checkpoints:
        step   = step_label(revision)
        result = _load_checkpoint(repo, revision, hf_token)
        if result is None:
            logger.warning(f"  SKIPPED step {step} ({revision}) — load failed.")
            continue
        model, tokenizer = result
        ckpt_vecs = _probe_vecs_for_checkpoint(model, tokenizer, probe_words)
        _free(model, tokenizer)

        if prev_vecs is not None:
            shared = sorted(set(prev_vecs.keys()) & set(ckpt_vecs.keys()))
            if len(shared) >= 2:
                X   = np.stack([prev_vecs[w]  for w in shared])
                Y   = np.stack([ckpt_vecs[w]  for w in shared])
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
# Batch runner
# ---------------------------------------------------------------------------

def _iso_codes_for_repo(repo: str) -> list[str]:
    return [code for code, models in MODEL_GROUPS.items() if repo in models]


def _lang_pair_codes(repo: str) -> list[str]:
    name  = repo.split("/")[-1].replace("beetlelm_", "")
    for tag in ("_mono", "_balanced", "_simultaneous", "_sequential",
                "_part_time", "_late", "_heritage"):
        name = name.replace(tag, "")
    name  = re.sub(r"_L[12]", "", name)
    codes = re.split(r"[-_]", name)
    known = set(FLORES_LANG_MAP.keys())
    return [c for c in codes if c in known]


def run_convergence_analysis(
    repos: list[str],
    hf_token: Optional[str] = None,
    overwrite: bool = False,
    signals: list[str] = ("ppl", "drift", "cka"),
) -> dict[str, dict[str, pd.DataFrame]]:
    results: dict[str, dict] = {}

    for repo in repos:
        iso_codes   = _iso_codes_for_repo(repo)
        lang_pair   = _lang_pair_codes(repo)
        primary_iso = iso_codes[0] if iso_codes else (lang_pair[0] if lang_pair else None)

        logger.info(f"\n{'='*60}")
        logger.info(f"Convergence: {repo}")
        logger.info(f"  ISO groups : {iso_codes}  |  PPL langs: {lang_pair}  |  Primary: {primary_iso}")

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

    # Concatenate all PPL trajectories
    ppl_frames = [v["ppl"] for v in results.values()
                  if "ppl" in v and isinstance(v["ppl"], pd.DataFrame) and not v["ppl"].empty]
    if ppl_frames:
        combined_path = OUTPUT_DIR / "all_ppl_trajectories.csv"
        pd.concat(ppl_frames, ignore_index=True).to_csv(combined_path, index=False)
        logger.info(f"\n[convergence] All PPL trajectories → {combined_path}")

    if CONVERGENCE_FAILURES:
        logger.info(f"\n[convergence] {'='*56}")
        logger.info(f"[convergence] SKIPPED {len(CONVERGENCE_FAILURES)} checkpoint(s) due to load errors:")
        for key, reason in CONVERGENCE_FAILURES.items():
            logger.info(f"  ✗  {key}")
            logger.info(f"     {reason}")
        logger.info(f"[convergence] {'='*56}")

    return results


# ---------------------------------------------------------------------------
# Focus repo sets for the key research questions
# ---------------------------------------------------------------------------

FOCUS_REPOS = {
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
    "rt_focus": [
        "BeetleLM/beetlelm_eng-nld_simultaneous",
        "BeetleLM/beetlelm_nld_L1-eng_L2_simultaneous",
        "BeetleLM/beetlelm_zho-eng_balanced",
        "BeetleLM/beetlelm_zho_L1-eng_L2_balanced",
    ],
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run checkpoint convergence analysis.")
    parser.add_argument("--repos",     nargs="+")
    parser.add_argument("--focus",     choices=list(FOCUS_REPOS.keys()), default="forgetting_focus")
    parser.add_argument("--signals",   nargs="+", choices=["ppl", "drift", "cka"],
                        default=["ppl", "drift", "cka"])
    parser.add_argument("--hf_token",  default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repos = args.repos if args.repos else FOCUS_REPOS[args.focus]
    run_convergence_analysis(repos, hf_token=args.hf_token,
                             overwrite=args.overwrite, signals=args.signals)