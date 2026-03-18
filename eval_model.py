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
  # single GPU debug
  python eval_model.py --rank 0 --world_size 1 --output_dir /path/to/beetle-analyze

  # one slice on GPU N of 8 (called by launch_all.sh)
  CUDA_VISIBLE_DEVICES=N python eval_model.py \\
      --rank N --world_size 8 --output_dir /path/to/beetle-analyze

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
# datasets reads HF_DATASETS_CACHE at import time, not at load_dataset() time.
# We set it here from --output_dir so it is always consistent, regardless of
# whether the parent shell exported HF_DATASETS_CACHE correctly.
def _pin_hf_cache() -> None:
    for i, arg in enumerate(sys.argv):
        if arg == "--output_dir" and i + 1 < len(sys.argv):
            cache = os.path.join(sys.argv[i + 1], "results", ".hf_cache")
            os.makedirs(cache, exist_ok=True)
            os.environ["HF_DATASETS_CACHE"] = cache
            os.environ["HF_HOME"]           = cache
            # Also expose the pkl cache dir so loaders can find it
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
from models import ALL_MODELS, MODEL_GROUPS, get_bilingual_type, get_lang_pair
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
# ═════════════════════════════════════════════════════════════════════════════

BENCHMARKS: Dict[str, dict] = {

    "multiblimp": dict(
        type            = "minimal_pair",
        hf_id           = "jumelet/multiblimp",
        config_mode     = "per_lang",      # load_dataset(hf_id, lang, split="train")
        good_col        = "sen",
        bad_col         = "wrong_sen",
        langs           = {"nld": "Dutch", "deu": "German", "fra": "French",
                           "fas": "Persian", "bul": "Bulgarian"},
        relevant_groups = {"nld": ["nld"], "deu": ["deu"], "fra": ["fra"],
                           "fas": ["fas"], "bul": ["bul"]},
    ),

    "blimp_eng": dict(
        type            = "minimal_pair",
        hf_id           = "nyu-mll/blimp",
        config_mode     = "all_configs",   # one config per phenomenon, same as zhoblimp/blimp_nl
        good_col        = "sentence_good",
        bad_col         = "sentence_bad",
        langs           = {"eng": "English"},
        relevant_groups = {"eng": ["eng"]},
    ),

    "zhoblimp": dict(
        type            = "minimal_pair",
        hf_id           = "Junrui1202/zhoblimp",
        config_mode     = "all_configs",   # one config per phenomenon — concat all
        good_col        = "sentence_good",
        bad_col         = "sentence_bad",
        langs           = {"zho": "Chinese"},
        relevant_groups = {"zho": ["zho"]},
    ),

    "blimp_nl": dict(
        type            = "minimal_pair",
        hf_id           = "juletxara/blimp-nl",
        config_mode     = "all_configs",
        good_col        = "sentence_good",
        bad_col         = "sentence_bad",
        langs           = {"nld": "Dutch"},
        relevant_groups = {"nld": ["nld"]},
    ),

    "xcomps": dict(
        type            = "minimal_pair",
        hf_id           = "fpadovani/xcomps-dataset",
        config_mode     = "split_per_lang",  # each lang is a named split
        good_col        = "acceptable_sent",
        bad_col         = "unacceptable_sent",
        langs           = {"fra": "French", "deu": "German", "ukr": "Ukrainian",
                           "zho": "Chinese", "fas": "Persian"},
        lang_split_map  = {"fra": "comps_fr", "deu": "comps_de", "ukr": "comps_uk",
                           "zho": "comps_zh", "fas": "comps_fa"},
        relevant_groups = {
            "fra": ["fra"], "deu": ["deu"], "zho": ["zho"], "fas": ["fas"],
            "ukr": ["nld", "deu", "zho", "fra", "fas", "bul"],
        },
    ),

    "xnli": dict(
        type            = "xnli",
        hf_id           = "xnli",
        config_mode     = "per_lang",       # load_dataset("xnli", lang, split="validation")
        langs           = {"en": "English", "fr": "French", "de": "German",
                           "zh": "Chinese", "bg": "Bulgarian"},
        relevant_groups = {"en": ["eng"], "fr": ["fra"], "de": ["deu"],
                           "zh": ["zho"], "bg": ["bul"]},
    ),
}

XNLI_SEP = " [SEP] "


# ═════════════════════════════════════════════════════════════════════════════
# DATASET LOADING
# All loaders del the HF Dataset object immediately after extracting tuples,
# so no Arrow tables remain in memory after the function returns.
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


# Maps hf_id → pkl filename written by prefetch_datasets.py
_PKL_MAP = {
    "Junrui1202/zhoblimp" : "zhoblimp.pkl",
    "juletxara/blimp-nl"  : "blimp_nl.pkl",
    "nyu-mll/blimp"       : "blimp_eng.pkl",
}


def _load_pairs_all_configs(hf_id: str, good: str, bad: str) -> List[Tuple[str, str]]:
    """
    Load from a pre-fetched pickle file if available (preferred — zero HTTP),
    otherwise fall back to live HF download with throttling.

    The pickle files are created by prefetch_datasets.py which is called once
    by launch_all.sh before any worker process starts.
    """
    import pickle

    pkl_dir  = os.environ.get("PKL_DIR", "")
    pkl_file = _PKL_MAP.get(hf_id, "")
    pkl_path = os.path.join(pkl_dir, pkl_file) if pkl_dir and pkl_file else ""

    if pkl_path and os.path.exists(pkl_path):
        with open(pkl_path, "rb") as fh:
            pairs = pickle.load(fh)
        logger.info(f"  Loaded {len(pairs):,} pairs from pkl cache ({pkl_file})")
        return pairs

    # ── Fallback: live download (should not normally be reached) ──────────
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
    Estimated total RAM: ~100 MB for all benchmarks combined.
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
    """Return the integer from a 'step-N' branch name, or None."""
    m = re.match(r"^step-(\d+)$", name)
    return int(m.group(1)) if m else None


def select_checkpoints(all_ckpts: List[str], latest_only: bool) -> List[str]:
    """
    If latest_only is True, return a single-element list containing only the
    checkpoint with the highest step number.  Falls back to ['main'] when no
    step-N branches exist.

    If latest_only is False, return all_ckpts unchanged (original behaviour).
    """
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
        del outputs              # free [B,T,V] logits BEFORE log_probs is used
                                 # otherwise both tensors live simultaneously

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
    # Binary: entailment (0) vs contradiction (2) — skip neutral (1)
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
# Each GPU process pushes independently after finishing one model.
# Only *.csv files are staged — the pkl/HF caches are never touched by git,
# so git never has to read or object-pack gigabytes of binary cache data.
# ═════════════════════════════════════════════════════════════════════════════

def git_push(repo_root: str, model_name: str, rank: int) -> None:
    """
    From repo_root:
        git add results/*.csv     (ONLY the CSV result files — not the caches)
        git commit -m "…"
        git push origin main      (retry up to 5× on non-fast-forward)

    Staging only CSVs prevents git from reading the pickle / HF cache
    directories into its object store, which previously caused an OOM crash.

    Multiple parallel processes may call this concurrently; git's remote
    ref-lock serialises the actual push. We handle the race with rebase+retry.
    """
    short = model_name.split("/")[-1]

    def run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, cwd=repo_root, capture_output=True,
                              text=True, check=check)

    try:
        # Stage ONLY the CSV result files.
        # Using shell glob expansion via subprocess requires shell=True, which
        # is a security risk; instead we enumerate the files in Python and pass
        # them explicitly so no shell is involved.
        import glob
        csv_files = glob.glob(os.path.join(repo_root, "results", "*.csv"))
        if not csv_files:
            logger.info(f"  git [{rank}]: no CSV files found to stage for {short}")
            return

        # git add takes multiple paths in one call — efficient and atomic.
        run(["git", "add", "--"] + csv_files)

        # Nothing staged? Nothing to do.
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
            # Non-fast-forward: rebase on top of what's on remote, then retry
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
    p.add_argument("--rank",        type=int, default=0,
                   help="Index of this process (0-indexed)")
    p.add_argument("--world_size",  type=int, default=1,
                   help="Total number of parallel processes")
    p.add_argument("--output_dir",  required=True,
                   help="Absolute path to the beetle-analyze repo root "
                        "(results/ will be created inside it)")
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--hf_token",    default=None)
    p.add_argument("--resume",      action="store_true", default=True)
    p.add_argument("--no_push",     action="store_true",
                   help="Skip git push (useful for dry runs)")
    p.add_argument("--latest_only", action="store_true",
                   help="Evaluate only the single highest step-N checkpoint per "
                        "model, instead of all checkpoints. Falls back to 'main' "
                        "when no step-N branches exist.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    # CUDA_VISIBLE_DEVICES is set by launch_all.sh to a single GPU, which
    # CUDA renumbers as cuda:0. We always use cuda:0 inside the process.
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
    repo_root  = os.path.abspath(args.output_dir)  # beetle-analyze/
    result_dir = os.path.join(repo_root, "results")
    os.makedirs(result_dir, exist_ok=True)

    logger.info(f"Rank         : {args.rank} / {args.world_size}")
    logger.info(f"Device       : {device}")
    logger.info(f"Repo root    : {repo_root}")
    logger.info(f"Results dir  : {result_dir}")
    logger.info(f"Latest only  : {args.latest_only}")

    # ── Model slice for this rank ─────────────────────────────────────────
    my_models = ALL_MODELS[args.rank :: args.world_size]
    logger.info(f"Models       : {len(my_models)} / {len(ALL_MODELS)} total\n")

    # ── Pre-load ALL benchmark data ───────────────────────────────────────
    # Runs once. All HF Dataset objects are deleted after pair extraction.
    # Total RAM for all benchmarks: ~100 MB as plain Python string lists.
    logger.info("Pre-loading benchmark datasets …")
    all_data = preload_all_datasets(logger)
    logger.info("Datasets ready.\n")

    # ── Helper: which (benchmark, lang) pairs apply to this model? ────────
    def applicable(repo: str) -> List[Tuple[str, str, str]]:
        """Returns [(bm_name, lang_code, csv_path), ...]"""
        group = next((g for g, ms in MODEL_GROUPS.items() if repo in ms), None)
        result = []
        for bm_name, cfg in BENCHMARKS.items():
            for lang_code, relevant in cfg["relevant_groups"].items():
                if group in relevant and lang_code in all_data.get(bm_name, {}):
                    csv_path = os.path.join(result_dir, f"{bm_name}_results.csv")
                    result.append((bm_name, lang_code, csv_path))
        return result

    # ══════════════════════════════════════════════════════════════════════
    # OUTER LOOP — one model repo at a time
    # ══════════════════════════════════════════════════════════════════════
    for repo in my_models:
        all_checkpoints = list_checkpoints(repo, args.hf_token)
        # ── Optionally restrict to the single highest step-N checkpoint ───
        checkpoints = select_checkpoints(all_checkpoints, args.latest_only)

        bil_type    = get_bilingual_type(repo)
        lang_pair   = get_lang_pair(repo)
        tasks       = applicable(repo)

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

            # Which tasks still need evaluation for this checkpoint?
            needed = []
            for bm_name, lang_code, csv_path in tasks:
                lang_name = BENCHMARKS[bm_name]["langs"][lang_code]
                if args.resume and already_done(csv_path, repo, ckpt, lang_name, bm_name):
                    pass  # already recorded, skip silently
                else:
                    needed.append((bm_name, lang_code, csv_path, lang_name))

            if not needed:
                logger.info(f"  {ckpt}: all done — skipping model load.")
                continue

            # ── Load model ONCE ───────────────────────────────────────────
            # low_cpu_mem_usage=True: weights allocated directly into their
            # final location — no extra CPU copy before .to(device).
            # Peak CPU RAM = ~1× model size, drops to ~0 after .to(device).
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

            # ── Score on every needed benchmark × language ─────────────────
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

            # ── FREE model — guaranteed before next checkpoint loads ───────
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"  {ckpt}: model freed.")

        # ── After all checkpoints for this model: push to GitHub ──────────
        if model_had_new_results and not args.no_push:
            git_push(repo_root, repo, args.rank)

    logger.info("\nAll models evaluated.")


if __name__ == "__main__":
    main()
