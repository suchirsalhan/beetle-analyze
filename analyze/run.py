"""
run_pipeline.py — Master runner for the BeetleLM analysis pipeline.

Usage examples
--------------
# Run everything (all languages, all analyses):
python run.py --all

# Just PPL + forgetting for deu/nld/zho:
python run.py --ppl --forgetting --langs deu nld zho

# Embedding drift for Chinese only:
python run.py --embeddings --langs zho

# Generate all figures from existing results:
python run.py --figures --langs deu nld zho

# Re-run everything, overwriting cached results:
python run.py --all --overwrite
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Submodule imports (lazy — only imported when needed)
# ---------------------------------------------------------------------------

def _run_ppl(args):
    from ppl_eval import run_all_languages
    print("\n" + "="*60 + "\n PPL EVALUATION\n" + "="*60)
    run_all_languages(lang_codes=args.langs, overwrite=args.overwrite)


def _run_forgetting(args):
    from forgetting import run_forgetting_analysis
    langs = args.langs or ["deu", "nld", "zho"]
    print("\n" + "="*60 + "\n CATASTROPHIC FORGETTING\n" + "="*60)
    run_forgetting_analysis(lang_codes=langs, overwrite=args.overwrite)


def _run_embeddings(args):
    from embedding_drift import run_embedding_analysis
    print("\n" + "="*60 + "\n EMBEDDING DRIFT\n" + "="*60)
    run_embedding_analysis(lang_codes=args.langs, overwrite=args.overwrite)


def _run_convergence(args):
    from convergence import run_convergence_analysis, FOCUS_REPOS

    print("\n" + "="*60 + "\n CONVERGENCE / CHECKPOINT DRIFT\n" + "="*60)

    if args.langs:
        from models import MODEL_GROUPS
        repos = [r for code in args.langs for r in MODEL_GROUPS.get(code, [])]
    else:
        repos = FOCUS_REPOS["forgetting_focus"]

    run_convergence_analysis(
        repos,
        hf_token  = getattr(args, "hf_token", None),
        overwrite = args.overwrite,
        signals   = ["ppl", "drift", "cka"],
    )


def _run_figures(args):
    import visualise as viz

    print("\n" + "="*60 + "\n GENERATING FIGURES\n" + "="*60)
    langs = args.langs or ["deu", "nld", "zho"]

    # Global plots (all languages)
    for fn_name in ["ppl_heatmap", "forgetting_barplot", "ppl_convergence_curves",
                    "cka_convergence_plot"]:
        fn = getattr(viz, fn_name)
        try:
            fn()
        except FileNotFoundError as e:
            print(f"[figures] {fn_name}: skipped — {e}")
        except Exception as e:
            print(f"[figures] {fn_name}: ERROR — {e}")

    # Per-language plots
    per_lang_fns = [
        "forgetting_scatter",
        "vocab_overlap_plot",
        "probe_pca_plot",
        "cka_heatmap",
        "condition_summary",
        "ppl_convergence_overlay",
        "drift_trajectory_plot",
        "forgetting_inflection_plot",
    ]
    for lang in langs:
        for fn_name in per_lang_fns:
            fn = getattr(viz, fn_name)
            try:
                fn(lang=lang)
            except FileNotFoundError as e:
                print(f"[figures] {fn_name}({lang}): skipped — {e}")
            except Exception as e:
                print(f"[figures] {fn_name}({lang}): ERROR — {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BeetleLM analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--langs", nargs="+",
        help="ISO-639-3 language codes to process (default: all)"
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-compute even if cached results exist"
    )

    # Analysis switches
    p.add_argument("--all",          action="store_true", help="Run all analyses")
    p.add_argument("--ppl",          action="store_true", help="Run PPL/FLORES evaluation")
    p.add_argument("--forgetting",   action="store_true", help="Run catastrophic forgetting")
    p.add_argument("--embeddings",   action="store_true", help="Run embedding drift analysis")
    p.add_argument("--convergence",  action="store_true",
                   help="Run checkpoint-level convergence + drift analysis "
                        "(requires step-N branches on HF hub)")
    p.add_argument("--hf_token",     default=None,
                   help="HuggingFace token (needed for private repos / checkpoint listing)")
    p.add_argument("--figures",      action="store_true", help="Generate figures from results")

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if not any([args.all, args.ppl, args.forgetting,
                args.embeddings, args.reading_time, args.convergence, args.figures]):
        parser.print_help()
        sys.exit(0)

    t0 = time.time()

    if args.all or args.ppl:
        _run_ppl(args)

    if args.all or args.forgetting:
        _run_forgetting(args)

    if args.all or args.embeddings:
        _run_embeddings(args)

    if args.all or args.reading_time:
        _run_reading_time(args)

    if args.all or args.convergence:
        _run_convergence(args)

    if args.all or args.figures:
        _run_figures(args)

    elapsed = time.time() - t0
    print(f"\n✓ Pipeline complete in {elapsed/60:.1f} min")
    print(f"  Outputs in: results/")


if __name__ == "__main__":
    main()
