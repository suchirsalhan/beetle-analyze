#!/usr/bin/env python3
"""
eval_minimal_pairs.py — Evaluate BeetleLM models on minimal-pair benchmarks.

Benchmarks
----------
  multiblimp   : jumelet/multiblimp            langs: nld deu fra fas bul
  blimp_eng    : nyu-mll/blimp                 langs: eng
  zhoblimp     : Junrui1202/zhoblimp           langs: zho
  blimp_nl     : juletxara/blimp-nl            langs: nld
  xcomps       : fpadovani/xcomps-dataset      langs: fra deu ukr zho fas

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

Resume support: already-evaluated (model, checkpoint, language) triples are
skipped automatically so you can re-run after a crash without duplicates.
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset, get_dataset_config_names

sys.path.insert(0, os.path.dirname(__file__))
from models import ALL_MODELS, MODEL_GROUPS, get_bilingual_type, get_lang_pair
from utils  import (
    list_checkpoints, load_model_and_tokenizer, release,
    minimal_pair_accuracy, append_result, already_done,
)

logging.basicConfig(
    level  = logging.INFO,
    format = "[%(asctime)s] [GPU%(gpu)s] %(levelname)s %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Benchmark configs ─────────────────────────────────────────────────────────
#
# config_mode controls how load_pairs() fetches data:
#
#   "per_lang"      — load_dataset(hf_id, lang_code, split="train")
#                     Used by MultiBLiMP which has one HF config per language.
#
#   "all_configs"   — fetch every named config, concat all pairs.
#                     Used by BLiMP-NL and ZhoBLiMP whose configs are phenomena
#                     (e.g. 'verb_second', 'BA_verb_le_b') with no top-level
#                     'train' split.
#
#   "split_per_lang" — load_dataset(hf_id, split=lang_split_map[lang_code])
#                     Used by XCOMPs whose splits are named 'comps_fr',
#                     'comps_de', 'comps_uk', 'comps_zh', 'comps_fa'.
#                     Requires a "lang_split_map" key in the config dict.
#
#   "single"        — load_dataset(hf_id, split="train")
#                     Used by BLiMP-eng which has a flat single-config layout.

BENCHMARK_CFG: Dict[str, dict] = {

    # ── MultiBLiMP ─────────────────────────────────────────────────────────
    "multiblimp": dict(
        hf_id           = "jumelet/multiblimp",
        config_mode     = "per_lang",
        good_col        = "sen",
        bad_col         = "wrong_sen",
        langs           = {
            "nld": "Dutch",
            "deu": "German",
            "fra": "French",
            "fas": "Persian",
            "bul": "Bulgarian",
        },
        relevant_groups = {
            "nld": ["nld"],
            "deu": ["deu"],
            "fra": ["fra"],
            "fas": ["fas"],
            "bul": ["bul"],
        },
    ),

    # ── BLiMP (English) ────────────────────────────────────────────────────
    # Flat layout: one config, split="train"
    "blimp_eng": dict(
        hf_id           = "nyu-mll/blimp",
        config_mode     = "single",
        good_col        = "sentence_good",
        bad_col         = "sentence_bad",
        langs           = {"eng": "English"},
        relevant_groups = {"eng": ["eng"]},
    ),

    # ── ZhoBLiMP ───────────────────────────────────────────────────────────
    # Configs are phenomenon names (BA_verb_le_b, passive_suo, …).
    # We load ALL configs and concatenate — this gives the full benchmark.
    "zhoblimp": dict(
        hf_id           = "Junrui1202/zhoblimp",
        config_mode     = "all_configs",
        good_col        = "sentence_good",
        bad_col         = "sentence_bad",
        langs           = {"zho": "Chinese"},
        relevant_groups = {"zho": ["zho"]},
    ),

    # ── BLiMP-NL ───────────────────────────────────────────────────────────
    # Configs are phenomenon names (adpositional_phrases, verb_second, …).
    # We load ALL configs and concatenate.
    "blimp_nl": dict(
        hf_id           = "juletxara/blimp-nl",
        config_mode     = "all_configs",
        good_col        = "sentence_good",
        bad_col         = "sentence_bad",
        langs           = {"nld": "Dutch"},
        relevant_groups = {"nld": ["nld"]},
    ),

    # ── XCOMPs ─────────────────────────────────────────────────────────────
    # Splits are named comps_fr, comps_de, comps_uk, comps_zh, comps_fa.
    # There is no "train" split at the top level.
    "xcomps": dict(
        hf_id           = "fpadovani/xcomps-dataset",
        config_mode     = "split_per_lang",
        good_col        = "acceptable_sent",
        bad_col         = "unacceptable_sent",
        langs           = {
            "fra": "French",
            "deu": "German",
            "ukr": "Ukrainian",
            "zho": "Chinese",
            "fas": "Persian",
        },
        # Maps our ISO code → the actual HF split name for this dataset
        lang_split_map  = {
            "fra": "comps_fr",
            "deu": "comps_de",
            "ukr": "comps_uk",
            "zho": "comps_zh",
            "fas": "comps_fa",
        },
        relevant_groups = {
            "fra": ["fra"],
            "deu": ["deu"],
            "ukr": ["nld", "deu", "zho", "fra", "fas", "bul"],
            "zho": ["zho"],
            "fas": ["fas"],
        },
    ),
}


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_pairs(cfg: dict, lang: str) -> List[Tuple[str, str]]:
    """
    Load minimal pairs for a benchmark × language combination.
    Handles all four config_mode values described in BENCHMARK_CFG.
    """
    hf_id    = cfg["hf_id"]
    mode     = cfg["config_mode"]
    good_col = cfg["good_col"]
    bad_col  = cfg["bad_col"]

    # ── per_lang: one HF config per ISO language code ─────────────────────
    if mode == "per_lang":
        ds = load_dataset(hf_id, lang, split="train")
        return list(zip(ds[good_col], ds[bad_col]))

    # ── single: flat dataset, single "train" split ────────────────────────
    elif mode == "single":
        ds = load_dataset(hf_id, split="train")
        return list(zip(ds[good_col], ds[bad_col]))

    # ── all_configs: phenomenon-per-config layout — concat everything ──────
    elif mode == "all_configs":
        config_names = get_dataset_config_names(hf_id)
        pairs = []
        for cfg_name in config_names:
            try:
                # Some phenomenon configs only have a default split
                try:
                    ds = load_dataset(hf_id, cfg_name, split="train")
                except Exception:
                    # fall back: take first available split
                    ds_dict = load_dataset(hf_id, cfg_name)
                    first_split = list(ds_dict.keys())[0]
                    ds = ds_dict[first_split]
                pairs.extend(zip(ds[good_col], ds[bad_col]))
            except Exception as e:
                logger.warning(f"    Skipping config '{cfg_name}': {e}")
        logger.info(f"  all_configs: loaded {len(pairs):,} pairs from "
                    f"{len(config_names)} configs")
        return pairs

    # ── split_per_lang: each language is a named split ────────────────────
    elif mode == "split_per_lang":
        split_name = cfg["lang_split_map"][lang]
        ds = load_dataset(hf_id, split=split_name)
        return list(zip(ds[good_col], ds[bad_col]))

    else:
        raise ValueError(f"Unknown config_mode: {mode!r}")


# ── Model selection ───────────────────────────────────────────────────────────

def models_for_benchmark(cfg: dict, rank: int, world_size: int) -> List[str]:
    seen  = set()
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
    p.add_argument("--benchmark",  required=True,
                   choices=list(BENCHMARK_CFG.keys()),
                   help="Which benchmark to run")
    p.add_argument("--gpu",        type=int, default=0)
    p.add_argument("--rank",       type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--output_dir", default="results")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hf_token",   default=None)
    p.add_argument("--resume",     action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    if torch.cuda.is_available():
        n_visible = torch.cuda.device_count()
        if args.gpu >= n_visible:
            raise RuntimeError(
                f"--gpu {args.gpu} is out of range: only {n_visible} GPU(s) visible "
                f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}). "
                f"If launching via launch_all.sh with CUDA_VISIBLE_DEVICES set per process, "
                f"pass --gpu 0 and let the env var do the device mapping."
            )
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    old_factory = logging.getLogRecordFactory()
    def record_factory(*a, **kw):
        rec = old_factory(*a, **kw)
        rec.gpu = args.gpu
        return rec
    logging.setLogRecordFactory(record_factory)

    benchmark = args.benchmark
    cfg       = BENCHMARK_CFG[benchmark]
    csv_path  = os.path.join(args.output_dir, f"{benchmark}_results.csv")
    my_models = models_for_benchmark(cfg, args.rank, args.world_size)

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
            langs_needed = []
            for lang in datasets:
                if args.resume and already_done(
                    csv_path, repo, ckpt, cfg["langs"][lang], benchmark
                ):
                    logger.info(f"  SKIP {repo}@{ckpt} / {lang} (already in CSV)")
                else:
                    langs_needed.append(lang)

            if not langs_needed:
                continue

            logger.info(f"  Loading {repo} @ {ckpt} …")
            try:
                model, tokenizer = load_model_and_tokenizer(
                    repo, ckpt, device, args.hf_token
                )
            except Exception as e:
                logger.error(f"  Load failed: {e}")
                continue

            for lang in langs_needed:
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
