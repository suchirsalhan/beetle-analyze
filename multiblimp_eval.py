import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, get_dataset_config_names
from tqdm.auto import tqdm
import pandas as pd
from huggingface_hub import HfApi
import re

# ============================
# CONFIG
# ============================
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

MODEL_REPOS = [
    # --- MONO ---
    "BeetleLM/beetlelm_eng_mono",
    "BeetleLM/beetlelm_nld_mono",
    "BeetleLM/beetlelm_zho_mono",
    "BeetleLM/beetlelm_deu_mono",
    "BeetleLM/beetlelm_fra_mono",
    "BeetleLM/beetlelm_fas_mono",
    "BeetleLM/beetlelm_bul_mono",
    "BeetleLM/beetlelm_ukr_mono",
    "BeetleLM/beetlelm_ind_mono",

    # --- BASIC BILINGUAL ---
    "BeetleLM/beetlelm_eng-nld_simultaneous",
    "BeetleLM/beetlelm_eng-nld_balanced",
    "BeetleLM/beetlelm_eng-nld_sequential",
    "BeetleLM/beetlelm_eng-nld_part_time",

    "BeetleLM/beetlelm_eng_L1-deu_L2_simultaneous",
    "BeetleLM/beetlelm_eng_L1-deu_L2_balanced",
    "BeetleLM/beetlelm_deu_L1-eng_L2_simultaneous",
    "BeetleLM/beetlelm_deu_L1-eng_L2_balanced",

    "BeetleLM/beetlelm_eng_L1-zho_L2_simultaneous",
    "BeetleLM/beetlelm_zho-eng_balanced",
    "BeetleLM/beetlelm_zho_L1-eng_L2_simultaneous",
    "BeetleLM/beetlelm_zho_L1-eng_L2_balanced",

    # --- TRILINGUAL CORE ---
    "BeetleLM/beetlelm_zho_L1-nld_L2-eng_L3_part_time",
    "BeetleLM/beetlelm_zho_L1-eng_L2-nld_L3_part_time",
    "BeetleLM/beetlelm_nld_L1-zho_L2-eng_L3_part_time",
    "BeetleLM/beetlelm_nld_L1-eng_L2-zho_L3_part_time",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_part_time",
    "BeetleLM/beetlelm_eng_L1-zho_L2-nld_L3_part_time",

    "BeetleLM/beetlelm_zho_L1-nld_L2-eng_L3_late",
    "BeetleLM/beetlelm_zho_L1-eng_L2-nld_L3_late",
    "BeetleLM/beetlelm_nld_L1-zho_L2-eng_L3_late",
    "BeetleLM/beetlelm_nld_L1-eng_L2-zho_L3_late",
    "BeetleLM/beetlelm_eng_L1-zho_L2-nld_L3_late",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_late",

    "BeetleLM/beetlelm_zho_L1-nld_L2-eng_L3_heritage",
    "BeetleLM/beetlelm_zho_L1-eng_L2-nld_L3_heritage",
    "BeetleLM/beetlelm_nld_L1-zho_L2-eng_L3_heritage",
    "BeetleLM/beetlelm_nld_L1-eng_L2-zho_L3_heritage",
    "BeetleLM/beetlelm_eng_L1-zho_L2-nld_L3_heritage",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_heritage",

    "BeetleLM/beetlelm_zho_L1-eng_L2-nld_L3_balanced",
    "BeetleLM/beetlelm_zho_L1-nld_L2-eng_L3_balanced",
    "BeetleLM/beetlelm_nld_L1-zho_L2-eng_L3_balanced",
    "BeetleLM/beetlelm_nld_L1-eng_L2-zho_L3_balanced",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_balanced",
    "BeetleLM/beetlelm_eng_L1-zho_L2-nld_L3_balanced",

    # --- BROADER BILINGUAL (SUPPORTED LANGS ONLY) ---
    "BeetleLM/beetlelm_zho-ukr_simultaneous",
    "BeetleLM/beetlelm_nld-ukr_simultaneous",
    "BeetleLM/beetlelm_zho-fas_simultaneous",
    "BeetleLM/beetlelm_fra-ind_simultaneous",
    "BeetleLM/beetlelm_nld-bul_simultaneous",
    "BeetleLM/beetlelm_zho-ind_simultaneous",
    "BeetleLM/beetlelm_fas-ind_simultaneous",
    "BeetleLM/beetlelm_fas-bul_simultaneous",
    "BeetleLM/beetlelm_zho-deu_simultaneous",
    "BeetleLM/beetlelm_nld-fra_simultaneous",
    "BeetleLM/beetlelm_bul-ukr_simultaneous",
    "BeetleLM/beetlelm_eng-fra_simultaneous",
    "BeetleLM/beetlelm_ind-deu_simultaneous",
    "BeetleLM/beetlelm_fas-ukr_simultaneous",
    "BeetleLM/beetlelm_zho-bul_simultaneous",
    "BeetleLM/beetlelm_zho-fra_simultaneous",
    "BeetleLM/beetlelm_fas-eng_simultaneous",
    "BeetleLM/beetlelm_nld-deu_simultaneous",
    "BeetleLM/beetlelm_fra-ukr_simultaneous",

    # --- SEQUENTIAL ---
    "BeetleLM/beetlelm_zho-fas_sequential",
    "BeetleLM/beetlelm_zho-ukr_sequential",
    "BeetleLM/beetlelm_zho-ind_sequential",
    "BeetleLM/beetlelm_zho-deu_sequential",
    "BeetleLM/beetlelm_nld-bul_sequential",
    "BeetleLM/beetlelm_nld-ukr_sequential",
    "BeetleLM/beetlelm_zho-fra_sequential",
    "BeetleLM/beetlelm_fas-ind_sequential",
    "BeetleLM/beetlelm_fra-ind_sequential",
    "BeetleLM/beetlelm_ind-deu_sequential",
    "BeetleLM/beetlelm_nld-fra_sequential",
    "BeetleLM/beetlelm_fas-bul_sequential",
    "BeetleLM/beetlelm_eng-fra_sequential",
    "BeetleLM/beetlelm_fas-ukr_sequential",
    "BeetleLM/beetlelm_fas-eng_sequential",
    "BeetleLM/beetlelm_nld-deu_sequential",
    "BeetleLM/beetlelm_zho-bul_sequential",
    "BeetleLM/beetlelm_eng-nld_sequential",
    "BeetleLM/beetlelm_eng-bul_sequential",

    # --- PART-TIME ---
    "BeetleLM/beetlelm_zho-ukr_part_time",
    "BeetleLM/beetlelm_zho-ind_part_time",
    "BeetleLM/beetlelm_nld-ukr_part_time",
    "BeetleLM/beetlelm_zho-fas_part_time",
    "BeetleLM/beetlelm_nld-fra_part_time",
    "BeetleLM/beetlelm_zho-deu_part_time",
    "BeetleLM/beetlelm_fra-ind_part_time",
    "BeetleLM/beetlelm_nld-bul_part_time",
    "BeetleLM/beetlelm_ind-deu_part_time",
    "BeetleLM/beetlelm_fas-ukr_part_time",
    "BeetleLM/beetlelm_zho-bul_part_time",
    "BeetleLM/beetlelm_zho-fra_part_time",
    "BeetleLM/beetlelm_fas-ind_part_time",
    "BeetleLM/beetlelm_fas-bul_part_time",
    "BeetleLM/beetlelm_fas-eng_part_time",
    "BeetleLM/beetlelm_eng-nld_part_time",
    "BeetleLM/beetlelm_fra-ukr_part_time",
    "BeetleLM/beetlelm_nld-deu_part_time",
    "BeetleLM/beetlelm_eng-fra_part_time",
    "BeetleLM/beetlelm_bul-ukr_part_time",
    "BeetleLM/beetlelm_fas-nld_part_time",
    "BeetleLM/beetlelm_fas-deu_part_time",
    "BeetleLM/beetlelm_bul-fra_part_time",
    "BeetleLM/beetlelm_eng-bul_part_time",
    "BeetleLM/beetlelm_deu-ukr_part_time",
    "BeetleLM/beetlelm_eng-ind_part_time",
]

# All ISO 639-3 codes present in MultiBLiMP (from the dataset documentation)
MULTIBLIMP_LANGS = {
    "abk", "aqz", "sqi", "amh", "grc", "hbo", "apu", "hye", "eus", "bel",
    "ben", "bho", "bor", "bre", "bul", "bua", "cat", "chu", "xcl", "ces",
    "dan", "nld", "egy", "eng", "myv", "est", "fao", "fin", "fra", "glg",
    "kat", "deu", "aln", "got", "guj", "heb", "azz", "hin", "hit", "hun",
    "isl", "gle", "ita", "quc", "xnr", "krl", "kxh", "kaz", "kir", "koi",
    "kpv", "lat", "lav", "lij", "lit", "olo", "nds", "mkd", "mar", "frm",
    "ell", "mdf", "yrl", "pcm", "kmr", "sme", "fro", "orv", "ota", "fas",
    "xpg", "pol", "por", "ron", "rus", "san", "gla", "hbs", "sms", "slk",
    "slv", "spa", "arb", "swe", "tam", "ttc", "tpn", "tur", "uig", "ukr",
    "hsb", "urd", "urb", "uzb", "vep", "wbp", "cym", "hyw", "wol", "sah",
    "nhi",
}

hf_token = None
api = HfApi()

# ============================
# PARSER (L1/L2/L3 + curriculum)
# ============================
def parse_model_metadata(repo_name):
    name = repo_name.split("/")[-1]

    # curriculum
    curriculum = None
    for c in ["balanced", "simultaneous", "sequential", "part_time", "late", "heritage"]:
        if c in name:
            curriculum = c
            break

    # explicit L1/L2/L3
    l1_match = re.search(r"([a-z]{3})_L1", name)
    l2_match = re.search(r"([a-z]{3})_L2", name)
    l3_match = re.search(r"([a-z]{3})_L3", name)

    L1 = l1_match.group(1) if l1_match else None
    L2 = l2_match.group(1) if l2_match else None
    L3 = l3_match.group(1) if l3_match else None

    # fallback: dash-separated format  e.g. beetlelm_zho-ukr_simultaneous
    base = name.replace("beetlelm_", "").split("_")[0]
    if "-" in base:
        parts = base.split("-")
        if L1 is None and len(parts) > 0:
            L1 = parts[0]
        if L2 is None and len(parts) > 1:
            L2 = parts[1]

    # mono fallback
    if L1 is None and "-" not in base:
        L1 = base

    return {"L1": L1, "L2": L2, "L3": L3, "curriculum": curriculum}


# ============================
# COLLECT ONLY NEEDED LANGUAGES
# ============================
print("Scanning model repos for referenced languages...")
needed_langs = set()
repo_metas = {}

for repo in MODEL_REPOS:
    meta = parse_model_metadata(repo)
    repo_metas[repo] = meta
    for lang in [meta["L1"], meta["L2"], meta["L3"]]:
        if lang is not None:
            needed_langs.add(lang)

# Intersect with what MultiBLiMP actually covers
available_in_multiblimp = get_dataset_config_names("jumelet/multiblimp")
langs_to_load = needed_langs & MULTIBLIMP_LANGS & set(available_in_multiblimp)

print(f"Languages referenced by models : {sorted(needed_langs)}")
print(f"Of those, covered by MultiBLiMP: {sorted(langs_to_load)}")
print(f"Not covered by MultiBLiMP      : {sorted(needed_langs - langs_to_load)}")

# ============================
# LOAD ONLY THE REQUIRED SUBSETS
# ============================
print("\nLoading MultiBLiMP splits...")
datasets = {}
for lang in sorted(langs_to_load):
    try:
        ds = load_dataset("jumelet/multiblimp", lang, split="train")
        datasets[lang] = list(zip(ds["sen"], ds["wrong_sen"]))
        print(f"  {lang}: {len(datasets[lang])} pairs")
    except Exception as e:
        print(f"  {lang}: FAILED ({e})")

print(f"\nLoaded {len(datasets)} language(s) total")

# ============================
# SCORING
# ============================
@torch.no_grad()
def score_sentences(model, tokenizer, sentences):
    enc = tokenizer(sentences, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    input_ids = input_ids.clamp(0, model.config.vocab_size - 1)

    outputs = model(input_ids)
    log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
    token_lp = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
    return token_lp.sum(dim=-1).cpu()


# ============================
# MAIN LOOP
# ============================
results = []

for repo in MODEL_REPOS:
    print(f"\n===== {repo} =====")
    meta = repo_metas[repo]

    # Only evaluate languages that are both in this model AND in MultiBLiMP
    eval_langs = [
        lang for lang in [meta["L1"], meta["L2"], meta["L3"]]
        if lang is not None and lang in datasets
    ]

    if not eval_langs:
        print("  Skipping — no model languages covered by MultiBLiMP")
        continue

    print(f"  Eval langs: {eval_langs}")

    # List checkpoints
    try:
        refs = api.list_repo_refs(repo_id=repo, token=hf_token)
        branches = [b.name for b in refs.branches]
        steps = sorted(
            [b for b in branches if re.match(r"^step-\d+$", b)],
            key=lambda x: int(x.split("-")[1])
        )
        ordered = steps + (["main"] if "main" in branches else [])
        print(f"  {len(ordered)} checkpoint(s) found")
    except Exception as e:
        print(f"  Failed to list refs: {e}")
        continue

    for branch in ordered:
        print(f"\n  → {branch}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                repo,
                revision=branch,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            ).to(DEVICE).eval()

            tokenizer = AutoTokenizer.from_pretrained(
                repo,
                revision=branch,
                trust_remote_code=True,
            )

            row = {
                "model":      repo,
                "checkpoint": branch,
                "L1":         meta["L1"],
                "L2":         meta["L2"],
                "L3":         meta["L3"],
                "curriculum": meta["curriculum"],
                "L1_acc":     None,
                "L2_acc":     None,
                "L3_acc":     None,
            }

            for lang in eval_langs:
                pairs   = datasets[lang]
                correct = 0

                for i in tqdm(range(0, len(pairs), BATCH_SIZE), desc=f"    {lang}"):
                    batch  = pairs[i : i + BATCH_SIZE]
                    good   = [p[0] for p in batch]
                    bad    = [p[1] for p in batch]
                    s_good = score_sentences(model, tokenizer, good)
                    s_bad  = score_sentences(model, tokenizer, bad)
                    correct += (s_good > s_bad).sum().item()

                acc = correct / len(pairs)
                print(f"    {lang}: {acc:.4f}")

                if lang == meta["L1"]:
                    row["L1_acc"] = acc
                elif lang == meta["L2"]:
                    row["L2_acc"] = acc
                elif lang == meta["L3"]:
                    row["L3_acc"] = acc

            results.append(row)

            del model, tokenizer
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  FAILED {repo} @ {branch}: {e}")
            continue

# ============================
# SAVE
# ============================
df = pd.DataFrame(results)
df.to_csv("multiblimp_results.csv", index=False)
print("\nSaved to multiblimp_results.csv")
