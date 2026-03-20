#!/usr/bin/env python3
"""
eval_model.py — Evaluate every BeetleLM model across all benchmarks,
                one model at a time to guarantee bounded RAM.

RAM contract (enforced in code, not just a promise)
----------------------------------------------------
At any moment during evaluation a single GPU process holds:

  Component                      When present        Size
  ─────────────────────────────  ──────────────────  ────────────────────────
  Benchmark pairs (plain lists)  always              ~100 MB total (all bms)
  Model weights (CPU, staging)   during .from_pret.  1× model  (float16)
  Model weights (GPU)            during scoring      1× model  (float16)
  Batch logits (GPU, transient)  one batch at a time [B,T,V] f16, del'd immed.

  CPU RAM peak  ≈  dataset_pairs  +  1× model_weights   (during load only)
  GPU RAM peak  ≈  1× model_weights  +  one_batch_logits

Every checkpoint loop ends with:
    del model, tokenizer  →  gc.collect()  →  torch.cuda.empty_cache()

The next iteration CANNOT start until that sequence completes, so there is
never more than one model in any form of memory at any time.

Usage
-----
  # single GPU debug (all bilingual models)
  python eval_model.py --rank 0 --world_size 1 --output_dir /path/to/beetle-analyze

  # one slice on GPU N of 8 (called by launch_all.sh)
  CUDA_VISIBLE_DEVICES=N python eval_model.py \\
      --rank N --world_size 8 --output_dir /path/to/beetle-analyze

  # trilingual models only (called by launch_trilingual.sh)
  CUDA_VISIBLE_DEVICES=N python eval_model.py \\
      --rank N --world_size 8 --output_dir /path/to/beetle-analyze --trilingual_only

  # evaluate only the highest step-N checkpoint per model
  CUDA_VISIBLE_DEVICES=N python eval_model.py \\
      --rank N --world_size 8 --output_dir /path/to/beetle-analyze --latest_only
"""

import argparse
import gc
import logging
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

# ── Pin HF cache BEFORE any HuggingFace import ───────────────────────────────
def _pin_hf_cache() -> None:
    for i, arg in enumerate(sys.argv):
        if arg == "--output_dir" and i + 1 < len(sys.argv):
            cache = os.path.join(sys.argv[i + 1], "results", ".hf_cache")
            os.makedirs(cache, exist_ok=True)
            os.environ["HF_DATASETS_CACHE"] = cache
            os.environ["HF_HOME"]           = cache
            os.environ["PKL_DIR"] = os.path.join(
                sys.argv[i + 1], "results", ".pkl_cache"
            )
            return
_pin_hf_cache()

import torch
import torch.nn.functional as F
from datasets import get_dataset_config_names, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import (
    ALL_MODELS, ALL_TRILINGUAL_MODELS, MODEL_GROUPS,
    get_bilingual_type, get_lang_pair,
)
from utils  import already_done, append_result, list_checkpoints

# ── Silence HF / datasets HTTP noise ─────────────────────────────────────────
for _lib in ("datasets", "huggingface_hub", "huggingface_hub.file_download",
             "fsspec", "httpx", "httpcore", "urllib3"):
    logging.getLogger(_lib).setLevel(logging.ERROR)

logging.basicConfig(
    level   = logging.INFO,
    format  = "[%(asctime)s] [rank%(rank)s] %(levelname)s %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# BENCHMARK DEFINITIONS
#
# relevant_groups maps each benchmark language to the MODEL_GROUPS keys that
# should be evaluated on it.  The "tri" key covers all trilingual eng–nld–zho
# models; they are tested on every benchmark that covers any of those three
# languages.
# ═════════════════════════════════════════════════════════════════════════════

BENCHMARKS: Dict[str, dict] = {

    "multiblimp": dict(
        type            = "minimal_pair",
        hf_id           = "jumelet/multiblimp",
        config_mode     = "per_lang",
        good_col        = "sen",
        bad_col         = "wrong_sen",
        langs           = {"nld": "Dutch", "deu": "German", "fra": "French",
                           "fas": "Persian", "bul": "Bulgarian"},
        # "tri" omitted: multiblimp has no eng or zho configs
        relevant_groups = {"nld": ["nld"], "deu": ["deu"], "fra": ["fra"],
                           "fas": ["fas"], "bul": ["bul"]},
    ),

    "blimp_eng": dict(
        type            = "minimal_pair",
        hf_id           = "nyu-mll/blimp",
        config_mode     = "all_configs",
        good_col        = "sentence_good",
        bad_col         = "sentence_bad",
        langs           = {"eng": "English"},
        # Trilingual models contain English → evaluate here too
        relevant_groups = {"eng": ["eng", "tri"]},
    ),

    "zhoblimp": dict(
        type            = "minimal_pair",
        hf_id           = "Junrui1202/zhoblimp",
        config_mode     = "all_configs",
        good_col        = "sentence_good",
        bad_col         = "sentence_bad",
        langs           = {"zho": "Chinese"},
        # Trilingual models contain Chinese → evaluate here too
        relevant_groups = {"zho": ["zho", "tri"]},
    ),

    "blimp_nl": dict(
        type            = "minimal_pair",
        hf_id           = "juletxara/blimp-nl",
        config_mode     = "all_configs",
        good_col        = "sentence_good",
        bad_col         = "sentence_bad",
        langs           = {"nld": "Dutch"},
        # Trilingual models contain Dutch → evaluate here too
        relevant_groups = {"nld": ["nld", "tri"]},
    ),

    "xcomps": dict(
        type            = "minimal_pair",
        hf_id           = "fpadovani/xcomps-dataset",
        config_mode     = "split_per_lang",
        good_col        = "acceptable_sent",
        bad_col         = "unacceptable_sent",
        langs           = {"fra": "French", "deu": "German", "ukr": "Ukrainian",
                           "zho": "Chinese", "fas": "Persian"},
        lang_split_map  = {"fra": "comps_fr", "deu": "comps_de", "ukr": "comps_uk",
                           "zho": "comps_zh", "fas": "comps_fa"},
        # Trilingual models contain Chinese → run xcomps/zho on them
        relevant_groups = {
            "fra": ["fra"], "deu": ["deu"], "zho": ["zho", "tri"],
            "fas": ["fas"],
            "ukr": ["nld", "deu", "zho", "fra", "fas", "bul"],
        },
    ),

    "xnli": dict(
        type            = "xnli",
        hf_id           = "xnli",
        config_mode     = "per_lang",
        langs           = {"en": "English", "fr": "French", "de": "German",
                           "zh": "Chinese", "bg": "Bulgarian"},
        # Trilingual models contain English and Chinese → run xnli/en and xnli/zh
        relevant_groups = {
            "en": ["eng", "tri"],
            "fr": ["fra"],
            "de": ["deu"],
            "zh": ["zho", "tri"],
            "bg": ["bul"],
        },
    ),
}

XNLI_SEP = " [SEP] "


# ═════════════════════════════════════════════════════════════════════════════
# DATASET LOADING
# ═════════════════════════════════════════════════════════════════════════════

def _load_pairs_single(hf_id: str, good: str, bad: str,
                       split: str = "train") -> List[Tuple[str, str]]:
    ds    = load_dataset(hf_id, split=split)
    pairs = list(zip(ds[good], ds[bad]))
    del ds
    return pairs


def _load_pairs_per_lang(hf_id: str, lang: str, good: str, bad: str,
                          split: str = "train") -> List[Tuple[str, str]]:
    ds    = load_dataset(hf_id, lang, split=split)
    pairs = list(zip(ds[good], ds[bad]))
    del ds
    return pairs


_PKL_MAP = {
    "Junrui1202/zhoblimp" : "zhoblimp.pkl",
    "juletxara/blimp-nl"  : "blimp_nl.pkl",
    "nyu-mll/blimp"       : "blimp_eng.pkl",
}


def _load_pairs_all_configs(hf_id: str, good: str, bad: str) -> List[Tuple[str, str]]:
    import pickle

    pkl_dir  = os.environ.get("PKL_DIR", "")
    pkl_file = _PKL_MAP.get(hf_id, "")
    pkl_path = os.path.join(pkl_dir, pkl_file) if pkl_dir and pkl_file else ""

    if pkl_path and os.path.exists(pkl_path):
        with open(pkl_path, "rb") as fh:
            pairs = pickle.load(fh)
        logger.info(f"  Loaded {len(pairs):,} pairs from pkl cache ({pkl_file})")
        return pairs

    logger.warning(
        f"  pkl cache not found for {hf_id} — falling back to live HF download. "
        f"Expected: {pkl_path or 'PKL_DIR not set'}"
    )
    configs = get_dataset_config_names(hf_id)
    pairs: List[Tuple[str, str]] = []
    for i, cfg_name in enumerate(configs):
        if i > 0:
            time.sleep(0.15)
        for attempt in range(4):
            try:
                try:
                    ds = load_dataset(hf_id, cfg_name, split="train")
                except Exception:
                    ds_dict = load_dataset(hf_id, cfg_name)
                    ds      = ds_dict[list(ds_dict.keys())[0]]
                pairs.extend(zip(ds[good], ds[bad]))
                del ds
                break
            except Exception as e:
                if "429" in str(e) and attempt < 3:
                    wait = 30 * (2 ** attempt)
                    logger.warning(f"    429 on '{cfg_name}' — waiting {wait}s")
                    time.sleep(wait)
                else:
                    logger.warning(f"    Skipping config '{cfg_name}': {e}")
                    break
        if (i + 1) % 25 == 0:
            logger.info(f"  … {i+1}/{len(configs)} configs loaded")
    gc.collect()
    return pairs


def _load_pairs_split_per_lang(hf_id: str, split_name: str,
                                good: str, bad: str) -> List[Tuple[str, str]]:
    ds    = load_dataset(hf_id, split=split_name)
    pairs = list(zip(ds[good], ds[bad]))
    del ds
    return pairs


def _load_xnli(lang: str) -> List[Tuple[str, str, int]]:
    ds      = load_dataset("xnli", lang, split="validation")
    triples = [(r["premise"], r["hypothesis"], r["label"]) for r in ds]
    del ds
    gc.collect()
    return triples


def preload_all_datasets(logger_) -> Dict[str, Dict[str, list]]:
    """
    Load every benchmark dataset into plain Python lists.
    No HF Dataset / Arrow objects survive after this function returns.
    """
    all_data: Dict[str, Dict[str, list]] = {}

    for bm_name, cfg in BENCHMARKS.items():
        all_data[bm_name] = {}
        mode = cfg["config_mode"]

        for lang_code, lang_name in cfg["langs"].items():
            logger_.info(f"  [{bm_name}] {lang_name} …")
            try:
                if cfg["type"] == "xnli":
                    data = _load_xnli(lang_code)
                elif mode == "per_lang":
                    data = _load_pairs_per_lang(
                        cfg["hf_id"], lang_code, cfg["good_col"], cfg["bad_col"]
                    )
                elif mode == "single":
                    data = _load_pairs_single(
                        cfg["hf_id"], cfg["good_col"], cfg["bad_col"]
                    )
                elif mode == "all_configs":
                    data = _load_pairs_all_configs(
                        cfg["hf_id"], cfg["good_col"], cfg["bad_col"]
                    )
                elif mode == "split_per_lang":
                    split = cfg["lang_split_map"][lang_code]
                    data = _load_pairs_split_per_lang(
                        cfg["hf_id"], split, cfg["good_col"], cfg["bad_col"]
                    )
                else:
                    raise ValueError(f"Unknown config_mode: {mode!r}")

                all_data[bm_name][lang_code] = data
                logger_.info(f"    {len(data):,} items")
            except Exception as e:
                logger_.warning(f"    Failed to load {bm_name}/{lang_code}: {e}")

    gc.collect()
    return all_data


# ═════════════════════════════════════════════════════════════════════════════
# CHECKPOINT SELECTION
# ═════════════════════════════════════════════════════════════════════════════

def _step_number(name: str) -> Optional[int]:
    m = re.match(r"^step-(\d+)$", name)
    return int(m.group(1)) if m else None


def select_checkpoints(all_ckpts: List[str], latest_only: bool) -> List[str]:
    if not latest_only:
        return all_ckpts

    step_ckpts = [(name, _step_number(name)) for name in all_ckpts
                  if _step_number(name) is not None]

    if not step_ckpts:
        logger.warning("  --latest_only: no step-N branches found, falling back to 'main'.")
        return ["main"]

    best = max(step_ckpts, key=lambda t: t[1])
    logger.info(f"  --latest_only: selected {best[0]} (step {best[1]:,}) "
                f"from {len(step_ckpts)} step checkpoints.")
    return [best[0]]


# ═════════════════════════════════════════════════════════════════════════════
# SCORING
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def score_sentences(model, tokenizer, sentences: List[str],
                    device: torch.device, batch_size: int) -> torch.Tensor:
    all_scores = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        enc   = tokenizer(batch, padding=True, truncation=True,
                          max_length=512, return_tensors="pt")
        input_ids      = enc["input_ids"].to(device).clamp(0, model.config.vocab_size - 1)
        attention_mask = enc["attention_mask"].to(device)

        outputs   = model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = F.log_softmax(outputs.logits, dim=-1)
        del outputs

        lp_shifted  = log_probs[:, :-1, :]
        ids_shifted = input_ids[:, 1:]
        mask_shifted= attention_mask[:, 1:].float()
        token_lp    = lp_shifted.gather(-1, ids_shifted.unsqueeze(-1)).squeeze(-1)
        sent_lp     = (token_lp * mask_shifted).sum(dim=-1)
        all_scores.append(sent_lp.cpu())

    return torch.cat(all_scores)


def run_minimal_pairs(model, tokenizer, pairs: List[Tuple[str, str]],
                      device: torch.device,
                      batch_size: int) -> Tuple[float, int, int]:
    s_good = score_sentences(model, tokenizer, [p[0] for p in pairs], device, batch_size)
    s_bad  = score_sentences(model, tokenizer, [p[1] for p in pairs], device, batch_size)
    n_correct = int((s_good > s_bad).sum().item())
    return n_correct / len(pairs) if pairs else 0.0, n_correct, len(pairs)


def run_xnli(model, tokenizer, triples: List[Tuple[str, str, int]],
             device: torch.device, batch_size: int) -> Tuple[float, int, int]:
    pairs = [(p, h, l) for p, h, l in triples if l in (0, 2)]
    pos   = [p + XNLI_SEP + h          for p, h, _ in pairs]
    neg   = [p + XNLI_SEP + "not " + h for p, h, _ in pairs]
    gold  = torch.tensor([l == 0 for _, __, l in pairs])
    s_pos = score_sentences(model, tokenizer, pos, device, batch_size)
    s_neg = score_sentences(model, tokenizer, neg, device, batch_size)
    correct = int(((s_pos > s_neg) == gold).sum().item())
    return correct / len(pairs) if pairs else 0.0, correct, len(pairs)


# ═════════════════════════════════════════════════════════════════════════════
# GIT PUSH
# ═════════════════════════════════════════════════════════════════════════════

def git_push(repo_root: str, model_name: str, rank: int) -> None:
    short = model_name.split("/")[-1]

    def run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, cwd=repo_root, capture_output=True,
                              text=True, check=check)

    try:
        import glob
        csv_files = glob.glob(os.path.join(repo_root, "results", "*.csv"))
        if not csv_files:
            logger.info(f"  git [{rank}]: no CSV files found to stage for {short}")
            return

        run(["git", "add", "--"] + csv_files)

        if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
            logger.info(f"  git [{rank}]: nothing new to commit for {short}")
            return

        run(["git", "commit", "-m", f"Add evaluation results: {short}"])

        for attempt in range(5):
            result = run(["git", "push", "origin", "main"], check=False)
            if result.returncode == 0:
                logger.info(f"  git [{rank}]: pushed {short}")
                return
            logger.warning(
                f"  git [{rank}]: push failed (attempt {attempt+1}/5) — "
                f"{result.stderr.strip()}"
            )
            run(["git", "pull", "--rebase", "origin", "main"], check=False)
            time.sleep(4 + attempt * 3)

        logger.error(f"  git [{rank}]: push gave up after 5 attempts for {short}")

    except subprocess.CalledProcessError as e:
        logger.error(f"  git [{rank}]: error — {e.stderr.strip()}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rank",            type=int, default=0)
    p.add_argument("--world_size",      type=int, default=1)
    p.add_argument("--output_dir",      required=True)
    p.add_argument("--batch_size",      type=int, default=64)
    p.add_argument("--hf_token",        default=None)
    p.add_argument("--resume",          action="store_true", default=True)
    p.add_argument("--no_push",         action="store_true")
    p.add_argument("--latest_only",     action="store_true",
                   help="Evaluate only the single highest step-N checkpoint per model.")
    p.add_argument("--trilingual_only", action="store_true",
                   help="Restrict evaluation to trilingual (eng–nld–zho) models only. "
                        "Uses ALL_TRILINGUAL_MODELS for rank slicing instead of ALL_MODELS.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        if torch.cuda.device_count() != 1:
            logger.warning(
                "Expected exactly 1 visible GPU (set by CUDA_VISIBLE_DEVICES) "
                f"but found {torch.cuda.device_count()}. Using cuda:0."
            )
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # ── Logging with rank ─────────────────────────────────────────────────
    old_factory = logging.getLogRecordFactory()
    rank_val    = args.rank

    def record_factory(*a, **kw):
        rec = old_factory(*a, **kw)
        rec.rank = rank_val
        return rec
    logging.setLogRecordFactory(record_factory)

    # ── Paths ─────────────────────────────────────────────────────────────
    repo_root  = os.path.abspath(args.output_dir)
    result_dir = os.path.join(repo_root, "results")
    os.makedirs(result_dir, exist_ok=True)

    logger.info(f"Rank            : {args.rank} / {args.world_size}")
    logger.info(f"Device          : {device}")
    logger.info(f"Repo root       : {repo_root}")
    logger.info(f"Results dir     : {result_dir}")
    logger.info(f"Latest only     : {args.latest_only}")
    logger.info(f"Trilingual only : {args.trilingual_only}")

    # ── Model slice for this rank ─────────────────────────────────────────
    # --trilingual_only slices from ALL_TRILINGUAL_MODELS so each GPU gets
    # a fair share of the 37 trilingual repos regardless of world_size.
    model_pool = ALL_TRILINGUAL_MODELS if args.trilingual_only else ALL_MODELS
    my_models  = model_pool[args.rank :: args.world_size]
    logger.info(f"Models          : {len(my_models)} / {len(model_pool)} "
                f"({'trilingual pool' if args.trilingual_only else 'full pool'})\n")

    # ── Pre-load ALL benchmark data ───────────────────────────────────────
    logger.info("Pre-loading benchmark datasets …")
    all_data = preload_all_datasets(logger)
    logger.info("Datasets ready.\n")

    # ── Helper: which (benchmark, lang) pairs apply to this model? ────────
    def applicable(repo: str) -> List[Tuple[str, str, str]]:
        """
        Return [(bm_name, lang_code, csv_path), ...] for every benchmark ×
        language that should be run on this repo.

        A model may belong to multiple MODEL_GROUPS (e.g. a trilingual repo
        sits in 'tri'; a bilingual repo may appear in overlapping lists).
        We collect ALL groups that contain the repo, then union the relevant
        benchmarks — avoiding duplicates with a seen set.
        """
        groups = {g for g, ms in MODEL_GROUPS.items() if repo in ms}
        seen   = set()
        result = []
        for bm_name, cfg in BENCHMARKS.items():
            for lang_code, relevant in cfg["relevant_groups"].items():
                if groups & set(relevant) and lang_code in all_data.get(bm_name, {}):
                    key = (bm_name, lang_code)
                    if key not in seen:
                        seen.add(key)
                        csv_path = os.path.join(result_dir, f"{bm_name}_results.csv")
                        result.append((bm_name, lang_code, csv_path))
        return result

    # ══════════════════════════════════════════════════════════════════════
    # OUTER LOOP — one model repo at a time
    # ══════════════════════════════════════════════════════════════════════
    for repo in my_models:
        all_checkpoints = list_checkpoints(repo, args.hf_token)
        checkpoints     = select_checkpoints(all_checkpoints, args.latest_only)

        bil_type  = get_bilingual_type(repo)
        lang_pair = get_lang_pair(repo)
        tasks     = applicable(repo)

        logger.info(f"{'='*60}")
        logger.info(f"Model    : {repo}")
        logger.info(f"Type     : {bil_type}  pair: {lang_pair}")
        logger.info(f"Ckpts    : {len(checkpoints)} "
                    f"({'of ' + str(len(all_checkpoints)) + ' total' if args.latest_only else 'total'})"
                    f"  tasks: {len(tasks)}")

        if not tasks:
            logger.warning("  No applicable benchmarks — skipping.")
            continue

        model_had_new_results = False

        # ── INNER LOOP — one checkpoint at a time ─────────────────────────
        for ckpt in checkpoints:

            needed = []
            for bm_name, lang_code, csv_path in tasks:
                lang_name = BENCHMARKS[bm_name]["langs"][lang_code]
                if args.resume and already_done(csv_path, repo, ckpt, lang_name, bm_name):
                    pass
                else:
                    needed.append((bm_name, lang_code, csv_path, lang_name))

            if not needed:
                logger.info(f"  {ckpt}: all done — skipping model load.")
                continue

            logger.info(f"  Loading @ {ckpt} …")
            try:
                load_kw = dict(
                    revision          = ckpt,
                    trust_remote_code = True,
                    torch_dtype       = torch.float16,
                    low_cpu_mem_usage = True,
                    token             = args.hf_token,
                )
                model = (
                    AutoModelForCausalLM
                    .from_pretrained(repo, **load_kw)
                    .to(device)
                    .eval()
                )
                tokenizer = AutoTokenizer.from_pretrained(repo, **load_kw)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            except Exception as e:
                logger.error(f"  Load failed: {e}")
                continue

            for bm_name, lang_code, csv_path, lang_name in needed:
                cfg  = BENCHMARKS[bm_name]
                data = all_data[bm_name][lang_code]
                try:
                    if cfg["type"] == "xnli":
                        acc, n_correct, n_total = run_xnli(
                            model, tokenizer, data, device, args.batch_size
                        )
                    else:
                        acc, n_correct, n_total = run_minimal_pairs(
                            model, tokenizer, data, device, args.batch_size
                        )
                    logger.info(
                        f"  {bm_name:12s}/{lang_name:10s} "
                        f"acc={acc:.4f}  ({n_correct}/{n_total})"
                    )
                    append_result(csv_path, {
                        "benchmark"     : bm_name,
                        "model"         : repo,
                        "lang_pair"     : lang_pair,
                        "bilingual_type": bil_type,
                        "checkpoint"    : ckpt,
                        "eval_language" : lang_name,
                        "accuracy"      : round(acc, 6),
                        "n_correct"     : n_correct,
                        "n_total"       : n_total,
                    })
                    model_had_new_results = True
                except Exception as e:
                    logger.error(f"  Eval failed [{bm_name}/{lang_code}]: {e}")

            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"  {ckpt}: model freed.")

        if model_had_new_results and not args.no_push:
            git_push(repo_root, repo, args.rank)

    logger.info("\nAll models evaluated.")


if __name__ == "__main__":
    main()
