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
import _path  # noqa: F401  — adds repo root to sys.path

from pathlib import Path

import pandas as pd

from ppl_utils import (
    GOLDFISH_TOKENS,
    LOAD_FAILURES,
    get_best_revision,
    load_flores_sentences,
    nll_to_ppl,
    score_sentences,
)
from models import MODEL_GROUPS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("results/ppl")


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

    Returns a DataFrame with columns
      [sentence_id, sentence, repo, revision, iso_code, nll, ppl].
    Skips repo/language combos that already have a saved CSV unless overwrite=True.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sentences = load_flores_sentences(iso_code)

    all_rows: list[dict] = []

    for repo in repos:
        slug     = repo.replace("/", "__")
        out_path = output_dir / f"ppl_{iso_code}_{slug}.csv"

        if out_path.exists() and not overwrite:
            print(f"[ppl_eval] Skipping {repo} ({iso_code}) — already exists.")
            all_rows.extend(pd.read_csv(out_path).to_dict("records"))
            continue

        revision = get_best_revision(repo)
        print(f"[ppl_eval] Scoring {iso_code} under {repo} @ {revision} …")
        nlls = score_sentences(sentences, repo, goldfish_tokens=goldfish_tokens)

        if nlls is None:
            print(f"[ppl_eval] SKIPPED {repo} — model failed to load.")
            continue

        rows = [
            {
                "sentence_id": i,
                "sentence":    sent,
                "repo":        repo,
                "revision":    revision,
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

    Returns summary DataFrame with columns
      [repo, revision, iso_code, mean_nll, mean_ppl, median_ppl, std_ppl].
    """
    if lang_codes is None:
        lang_codes = list(MODEL_GROUPS.keys())

    all_frames: list[pd.DataFrame] = []
    for code in lang_codes:
        df = evaluate_language(code, MODEL_GROUPS[code], overwrite=overwrite)
        all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()

    full = pd.concat(all_frames, ignore_index=True)

    summary = (
        full
        .groupby(["repo", "revision", "iso_code"])
        .agg(
            mean_nll   = ("nll", "mean"),
            mean_ppl   = ("ppl", "mean"),
            median_ppl = ("ppl", "median"),
            std_ppl    = ("ppl", "std"),
            n_sentences= ("nll", "count"),
        )
        .reset_index()
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "ppl_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n[ppl_eval] Summary saved → {summary_path}")

    if LOAD_FAILURES:
        print(f"\n[ppl_eval] {'='*56}")
        print(f"[ppl_eval] SKIPPED {len(LOAD_FAILURES)} model(s) due to load errors:")
        for repo, reason in LOAD_FAILURES.items():
            print(f"  ✗  {repo}")
            print(f"     {reason}")
        print(f"[ppl_eval] {'='*56}")

    return summary


# ---------------------------------------------------------------------------
# Convenience: score a single sentence interactively
# ---------------------------------------------------------------------------

def score_one(
    sentence: str,
    repo: str,
    goldfish_tokens: int | None = GOLDFISH_TOKENS,
) -> dict:
    """Quick helper for interactive / notebook use."""
    from ppl_utils import load_model_and_tokenizer, sentence_log_likelihood
    model, tokenizer = load_model_and_tokenizer(repo)
    nll = sentence_log_likelihood(sentence, model, tokenizer, goldfish_tokens)
    return {
        "repo":     repo,
        "revision": get_best_revision(repo),
        "sentence": sentence,
        "nll":      nll,
        "ppl":      nll_to_ppl(nll),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FLORES Goldfish PPL evaluation.")
    parser.add_argument("--langs", nargs="+", help="ISO codes to evaluate (default: all)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = run_all_languages(lang_codes=args.langs, overwrite=args.overwrite)
    print(summary.to_string())