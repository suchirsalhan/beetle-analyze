"""
ppl_eval.py — FLORES Goldfish PPL evaluation.

For each language, scores every sentence in the FLORES devtest split under
every model in the corresponding MODEL_GROUPS entry.

Outputs
-------
results/ppl/
  ppl_<lang>_<model_slug>.csv   — per-sentence NLL + PPL
  ppl_summary.csv               — mean PPL per (model, language)
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pandas as pd

from utils import (
    GOLDFISH_TOKENS,
    load_flores_sentences,
    nll_to_ppl,
    score_sentences,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("results/ppl")

# Languages to evaluate and which model groups cover them.
# Each entry: iso_code -> list of repos to score against that language.
# Import from models.py to stay DRY.
import sys
sys.path.insert(0, str(Path(__file__).parent))
from models import MODEL_GROUPS


# ---------------------------------------------------------------------------
# Per-language evaluation
# ---------------------------------------------------------------------------

def evaluate_language(
    iso_code: str,
    repos: list[str],
    output_dir: Path = OUTPUT_DIR,
    goldfish_tokens: int | None = GOLDFISH_TOKENS,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Score all FLORES sentences for *iso_code* under every repo in *repos*.

    Returns a DataFrame with columns [sentence_id, sentence, repo, nll, ppl].
    Skips repo/language combos that already have a saved CSV unless overwrite=True.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sentences = load_flores_sentences(iso_code)

    all_rows: list[dict] = []

    for repo in repos:
        slug = repo.replace("/", "__")
        out_path = output_dir / f"ppl_{iso_code}_{slug}.csv"

        if out_path.exists() and not overwrite:
            print(f"[ppl_eval] Skipping {repo} ({iso_code}) — already exists.")
            df = pd.read_csv(out_path)
            all_rows.extend(df.to_dict("records"))
            continue

        print(f"[ppl_eval] Scoring {iso_code} under {repo} …")
        nlls = score_sentences(sentences, repo, goldfish_tokens=goldfish_tokens)

        rows = [
            {
                "sentence_id": i,
                "sentence":    sent,
                "repo":        repo,
                "iso_code":    iso_code,
                "nll":         nll,
                "ppl":         nll_to_ppl(nll),
            }
            for i, (sent, nll) in enumerate(zip(sentences, nlls))
        ]
        pd.DataFrame(rows).to_csv(out_path, index=False)
        all_rows.extend(rows)
        print(f"  → saved {out_path}")

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Summary across all languages
# ---------------------------------------------------------------------------

def run_all_languages(
    lang_codes: list[str] | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Run PPL evaluation for all (or a subset of) languages.

    Args:
        lang_codes: ISO codes to evaluate. Defaults to all in MODEL_GROUPS.
        overwrite:  Re-score even if CSVs already exist.

    Returns:
        Summary DataFrame with columns [repo, iso_code, mean_nll, mean_ppl,
        median_ppl, std_ppl].
    """
    if lang_codes is None:
        lang_codes = list(MODEL_GROUPS.keys())

    all_frames: list[pd.DataFrame] = []

    for code in lang_codes:
        repos = MODEL_GROUPS[code]
        df = evaluate_language(code, repos, overwrite=overwrite)
        all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()

    full = pd.concat(all_frames, ignore_index=True)

    # Aggregate summary
    summary = (
        full
        .groupby(["repo", "iso_code"])
        .agg(
            mean_nll  = ("nll",  "mean"),
            mean_ppl  = ("ppl",  "mean"),
            median_ppl= ("ppl",  "median"),
            std_ppl   = ("ppl",  "std"),
            n_sentences=("nll",  "count"),
        )
        .reset_index()
    )

    summary_path = OUTPUT_DIR / "ppl_summary.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"\n[ppl_eval] Summary saved → {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# Convenience: score a single sentence interactively
# ---------------------------------------------------------------------------

def score_one(sentence: str, repo: str, goldfish_tokens: int | None = GOLDFISH_TOKENS) -> dict:
    """Quick helper for interactive/notebook use."""
    from utils import load_model_and_tokenizer, sentence_log_likelihood, nll_to_ppl
    model, tokenizer = load_model_and_tokenizer(repo)
    nll = sentence_log_likelihood(sentence, model, tokenizer, goldfish_tokens)
    return {"repo": repo, "sentence": sentence, "nll": nll, "ppl": nll_to_ppl(nll)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FLORES Goldfish PPL evaluation.")
    parser.add_argument("--langs", nargs="+", help="ISO codes to evaluate (default: all)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = run_all_languages(lang_codes=args.langs, overwrite=args.overwrite)
    print(summary.to_string())
