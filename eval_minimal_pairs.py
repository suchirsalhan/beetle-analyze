#!/usr/bin/env python3
"""
eval_minimal_pairs.py — Evaluate BeetleLM models on minimal-pair benchmarks.

Benchmarks
----------
  multiblimp   : jumelet/multiblimp          langs: nld deu fra fas bul
  zhoblimp     : … (see BENCHMARK_CFG)        langs: zho
  blimp_nl     : …                            langs: nld
  xcomps       : …                            langs: fra deu ukr zho fas

Usage — single GPU (test / debug)
----------------------------------
  python eval_minimal_pairs.py \\
      --benchmark multiblimp \\
      --gpu 0 \\
      --output_dir results/

Usage — one slice of models on GPU N out of 8
---------------------------------------------
  python eval_minimal_pairs.py \\
      --benchmark multiblimp \\
      --gpu 3 \\
      --rank 3 --world_size 8 \\
      --output_dir results/

The script is designed to be launched in parallel by launch_all.sh.
Each process independently picks up models[rank::world_size] so there is
no inter-process communication needed.

Resume support: already-evaluated (model, checkpoint, language) triples are
skipped automatically so you can re-run after a crash without duplicates.
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm

# local imports (same directory)
sys.path.insert(0, os.path.dirname(__file__))
from models import ALL_MODELS, MODEL_GROUPS, get_bilingual_type, get_lang_pair
from utils  import (
    list_checkpoints, load_model_and_tokenizer, release,
    minimal_pair_accuracy, append_result, already_done,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "[%(asctime)s] [GPU%(gpu)s] %(levelname)s %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Benchmark configs ─────────────────────────────────────────────────────────
#
# Each entry: hf_dataset_id, config_key_field, good_col, bad_col, langs_to_eval
#
# For datasets that use a language code as the config name:
#   config = lang  →  load_dataset(hf_id, lang)
# For datasets with a single config:
#   config = None  →  load_dataset(hf_id)
#
# langs_to_eval maps the ISO code used in the dataset to a display name.
# Only models in MODEL_GROUPS that include that ISO code will be evaluated on
# that language (filtering is done at runtime).

BENCHMARK_CFG: Dict[str, dict] = {
    # ── MultiBLiMP ─────────────────────────────────────────────────────────
    "multiblimp": dict(
        hf_id       = "jumelet/multiblimp",
        config_mode = "per_lang",      # load_dataset(hf_id, lang_code)
        good_col    = "sen",
        bad_col     = "wrong_sen",
        split       = "train",
        langs       = {
            "nld": "Dutch",
            "deu": "German",
            "fra": "French",
            "fas": "Persian",
            "bul": "Bulgarian",
        },
        # Which model groups to run for each language
        # (all models that are in that group, regardless of other lang in pair)
        relevant_groups = {
            "nld": ["nld"],
            "deu": ["deu"],
            "fra": ["fra"],
            "fas": ["fas"],
            "bul": ["bul"],
        },
    ),

  "blimp_eng": dict(
    hf_id       = "nyu-mll/blimp",       # Hugging Face dataset UID
    config_mode = "single",              # no per-lang subconfigs
    good_col    = "sentence_good",
    bad_col     = "sentence_bad",
    split       = "train",               # adjust if needed
    langs       = {"eng": "English"},    # ISO code → readable name
    relevant_groups = {"eng": ["eng"]},  # relevant group per language
    extra_fields = {
        "field": "field",
        "linguistics_term": "linguistics_term",
        "UID": "UID",
        "simple_LM_method": "simple_LM_method",
        "one_prefix_method": "one_prefix_method",
        "two_prefix_method": "two_prefix_method",
        "lexically_identical": "lexically_identical",
        "pair_id": "pair_id",
    },
            # Which model groups to run for each language
        # (all models that are in that group, regardless of other lang in pair)
        relevant_groups = {
            "eng": ["eng"],
        },
)

  # ── ZhoBLiMP ───────────────────────────────────────────────────────────
  "zhoblimp": dict(
      hf_id       = "Junrui1202/zhoblimp",    # updated HF dataset UID
      config_mode = "single",                  # no per-lang config
      good_col    = "sentence_good",           # matches dataset schema
      bad_col     = "sentence_bad",
      split       = "train",
      langs       = {"zho": "Chinese"},        # ISO code → readable name
      relevant_groups = {"zho": ["zho"]},      # groups per language
  ),

    # ── BLiMP-NL ───────────────────────────────────────────────────────────
    "blimp_nl": dict(
    hf_id       = "juletxara/blimp-nl",   # Hugging Face dataset UID
    config_mode = "single",               # no per-lang subconfigs
    good_col    = "sentence_good",
    bad_col     = "sentence_bad",
    split       = "train",                # or becomes dynamic if needed
    langs       = {"nld": "Dutch"},       # ISO code for Dutch
    relevant_groups = {"nld": ["nld"]},   # group names per language
    extra_fields = {
        "phenomenon": "linguistic_phenomenon",
        "paradigm": "paradigm",
        "item_id": "item_id",
        "critical_word": "critical_word",
        "cue_word": "cue_word",
    },
                  # Which model groups to run for each language
        # (all models that are in that group, regardless of other lang in pair)
        relevant_groups = {
            "nld": ["nld"],
        },
),

    # ── XCOMPs ─────────────────────────────────────────────────────────────
    "xcomps": dict(
        hf_id       = "fpadovani/xcomps-dataset",       # update if different
        config_mode = "single",                             # single config for all languages
        good_col    = "acceptable_sent",
        bad_col     = "unacceptable_sent",
        split       = "train",
        langs       = {
            "fra": "French",
            "deu": "German",
            "ukr": "Ukrainian",
            "zho": "Chinese",
            "fas": "Persian",
        },
        relevant_groups = {
            "fra": ["fra"],
            "deu": ["deu"],
            "ukr": ["nld", "deu", "zho", "fra", "fas", "bul"],  # all groups with Ukrainian models
            "zho": ["zho"],
            "fas": ["fas"],
        },
    ),
}


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_pairs(cfg: dict, lang: str) -> List[Tuple[str, str]]:
    """Load minimal pairs for a benchmark × language combination."""
    hf_id    = cfg["hf_id"]
    mode     = cfg["config_mode"]
    good_col = cfg["good_col"]
    bad_col  = cfg["bad_col"]
    split    = cfg["split"]

    if mode == "per_lang":
        ds = load_dataset(hf_id, lang, split=split)
    else:
        ds = load_dataset(hf_id, split=split)

    return list(zip(ds[good_col], ds[bad_col]))


# ── Model selection ───────────────────────────────────────────────────────────

def models_for_benchmark(cfg: dict, rank: int, world_size: int) -> List[str]:
    """
    Collect all model repos relevant to any language in this benchmark,
    deduplicate, then return the slice assigned to this rank.
    """
    seen = set()
    repos = []
    for lang, groups in cfg["relevant_groups"].items():
        for g in groups:
            for m in MODEL_GROUPS.get(g, []):
                if m not in seen:
                    seen.add(m)
                    repos.append(m)
    return repos[rank::world_size]


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark",   required=True,
                   choices=list(BENCHMARK_CFG.keys()),
                   help="Which benchmark to run")
    p.add_argument("--gpu",         type=int, default=0,
                   help="Which GPU index to use")
    p.add_argument("--rank",        type=int, default=0,
                   help="This process's rank (0-indexed)")
    p.add_argument("--world_size",  type=int, default=1,
                   help="Total number of parallel processes (= number of GPUs)")
    p.add_argument("--output_dir",  default="results",
                   help="Directory to write CSV files")
    p.add_argument("--batch_size",  type=int, default=64,
                   help="Tokenisation batch size (reduce if OOM)")
    p.add_argument("--hf_token",    default=None,
                   help="HuggingFace token for gated repos")
    p.add_argument("--resume",      action="store_true", default=True,
                   help="Skip already-completed (model, checkpoint, lang) triples")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Patch logger with GPU id
    old_factory = logging.getLogRecordFactory()
    def record_factory(*a, **kw):
        rec = old_factory(*a, **kw)
        rec.gpu = args.gpu
        return rec
    logging.setLogRecordFactory(record_factory)

    benchmark   = args.benchmark
    cfg         = BENCHMARK_CFG[benchmark]
    csv_path    = os.path.join(args.output_dir, f"{benchmark}_results.csv")
    my_models   = models_for_benchmark(cfg, args.rank, args.world_size)

    logger.info(f"Benchmark    : {benchmark}")
    logger.info(f"GPU          : {args.gpu}  rank {args.rank}/{args.world_size}")
    logger.info(f"Models slice : {len(my_models)}")
    logger.info(f"Output CSV   : {csv_path}")

    # Pre-load all datasets (small, fits in RAM)
    datasets: Dict[str, List[Tuple[str, str]]] = {}
    for lang, lang_name in cfg["langs"].items():
        try:
            logger.info(f"Loading {benchmark} / {lang_name} …")
            datasets[lang] = load_pairs(cfg, lang)
            logger.info(f"  {len(datasets[lang]):,} pairs")
        except Exception as e:
            logger.warning(f"  Failed to load {lang}: {e}")

    if not datasets:
        logger.error("No datasets loaded — exiting.")
        return

    # ── Evaluate ─────────────────────────────────────────────────────────────
    for repo in my_models:
        checkpoints = list_checkpoints(repo, args.hf_token)
        logger.info(f"\n{'='*60}")
        logger.info(f"Model : {repo}  ({len(checkpoints)} checkpoints)")

        bil_type  = get_bilingual_type(repo)
        lang_pair = get_lang_pair(repo)

        for ckpt in checkpoints:
            # Check if any language still needs evaluation for this checkpoint
            langs_needed = []
            for lang in datasets:
                if args.resume and already_done(csv_path, repo, ckpt, cfg["langs"][lang], benchmark):
                    logger.info(f"  SKIP {repo}@{ckpt} / {lang} (already in CSV)")
                else:
                    langs_needed.append(lang)

            if not langs_needed:
                continue

            # Load model once per checkpoint, score all needed languages
            logger.info(f"  Loading {repo} @ {ckpt} …")
            try:
                model, tokenizer = load_model_and_tokenizer(
                    repo, ckpt, device, args.hf_token
                )
            except Exception as e:
                logger.error(f"  Load failed: {e}")
                continue

            for lang in langs_needed:
                # Only evaluate if this language is in the relevant groups for this model
                relevant_groups = cfg["relevant_groups"].get(lang, [])
                model_group     = next(
                    (g for g, ms in MODEL_GROUPS.items() if repo in ms), None
                )
                if model_group not in relevant_groups:
                    continue

                lang_name = cfg["langs"][lang]
                pairs     = datasets[lang]

                try:
                    acc, n_correct, n_total = minimal_pair_accuracy(
                        model, tokenizer, pairs, device, args.batch_size
                    )
                    logger.info(
                        f"  {lang_name:12s}  acc={acc:.4f}  ({n_correct}/{n_total})"
                    )
                    append_result(csv_path, {
                        "benchmark"     : benchmark,
                        "model"         : repo,
                        "lang_pair"     : lang_pair,
                        "bilingual_type": bil_type,
                        "checkpoint"    : ckpt,
                        "eval_language" : lang_name,
                        "accuracy"      : round(acc, 6),
                        "n_correct"     : n_correct,
                        "n_total"       : n_total,
                    })
                except Exception as e:
                    logger.error(f"  Eval failed for {lang_name}: {e}")

            release(model, tokenizer, device)

    logger.info(f"\nDone. Results written to {csv_path}")


if __name__ == "__main__":
    main()
