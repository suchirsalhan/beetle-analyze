"""
models.py — Central registry of all BeetleLM model repos.

MODEL_GROUPS       : dict  lang_code -> list of repo strings
ALL_MODELS         : list  deduplicated across all groups, stable order
TRILINGUAL_MODELS  : list  all trilingual repos (balanced + sequential variants)
ALL_TRILINGUAL_MODELS : list  deduplicated trilingual-only list for rank slicing
"""

import re

DUTCH_MODELS = [
    "BeetleLM/beetlelm_nld_mono",
    # Balanced
    "BeetleLM/beetlelm_eng-nld_balanced",
    "BeetleLM/beetlelm_nld_L1-eng_L2_balanced",
    "BeetleLM/beetlelm_nld-ukr_balanced",
    "BeetleLM/beetlelm_nld-fra_balanced",
    "BeetleLM/beetlelm_nld-deu_balanced",
    "BeetleLM/beetlelm_nld-ind_balanced",
    "BeetleLM/beetlelm_nld-bul_balanced",
    "BeetleLM/beetlelm_zho-nld_balanced",
    # Simultaneous
    "BeetleLM/beetlelm_eng-nld_simultaneous",
    "BeetleLM/beetlelm_nld_L1-eng_L2_simultaneous",
    "BeetleLM/beetlelm_fas-nld_simultaneous",
    # Sequential
    "BeetleLM/beetlelm_fas-nld_sequential",
    # Part-Time
    "BeetleLM/beetlelm_eng-nld_part_time",
    "BeetleLM/beetlelm_fas-nld_part_time",
    "BeetleLM/beetlelm_nld-deu_part_time",
    # Late
    "BeetleLM/beetlelm_nld-ukr_late",
    "BeetleLM/beetlelm_nld-bul_late",
    "BeetleLM/beetlelm_fas-nld_late",
    "BeetleLM/beetlelm_eng-nld_late",
    # Heritage
    "BeetleLM/beetlelm_nld-fra_heritage",
    "BeetleLM/beetlelm_nld-deu_heritage",
    "BeetleLM/beetlelm_nld-bul_heritage",
    "BeetleLM/beetlelm_nld-ukr_heritage",
    "BeetleLM/beetlelm_fas-nld_heritage",
]

GERMAN_MODELS = [
    "BeetleLM/beetlelm_deu_mono",
    # Balanced
    "BeetleLM/beetlelm_deu_L1-eng_L2_balanced",
    "BeetleLM/beetlelm_eng_L1-deu_L2_balanced",
    # Simultaneous
    "BeetleLM/beetlelm_deu_L1-eng_L2_simultaneous",
    "BeetleLM/beetlelm_eng_L1-deu_L2_simultaneous",
    "BeetleLM/beetlelm_eng-deu_simultaneous",
    "BeetleLM/beetlelm_deu-ukr_simultaneous",
    "BeetleLM/beetlelm_bul-deu_simultaneous",
    # Sequential
    "BeetleLM/beetlelm_deu-ukr_sequential",
    "BeetleLM/beetlelm_bul-deu_sequential",
    # Part-Time
    "BeetleLM/beetlelm_nld-deu_part_time",
    "BeetleLM/beetlelm_deu-ukr_part_time",
    "BeetleLM/beetlelm_bul-deu_part_time",
    # Late
    "BeetleLM/beetlelm_zho-deu_late",
    "BeetleLM/beetlelm_fas-deu_late",
    "BeetleLM/beetlelm_deu-ukr_late",
    "BeetleLM/beetlelm_bul-deu_late",
    # Heritage
    "BeetleLM/beetlelm_zho-deu_heritage",
    "BeetleLM/beetlelm_ind-deu_heritage",
    "BeetleLM/beetlelm_fra-deu_heritage",
    "BeetleLM/beetlelm_fas-deu_heritage",
    "BeetleLM/beetlelm_eng-deu_heritage",
    "BeetleLM/beetlelm_deu-ukr_heritage",
    "BeetleLM/beetlelm_bul-deu_heritage",
]

CHINESE_MODELS = [
    "BeetleLM/beetlelm_zho_mono",
    # Balanced
    "BeetleLM/beetlelm_zho_L1-eng_L2_balanced",
    "BeetleLM/beetlelm_zho-eng_balanced",
    "BeetleLM/beetlelm_zho-ukr_balanced",
    "BeetleLM/beetlelm_zho-fra_balanced",
    "BeetleLM/beetlelm_zho-fas_balanced",
    "BeetleLM/beetlelm_zho-ind_balanced",
    "BeetleLM/beetlelm_zho-deu_balanced",
    "BeetleLM/beetlelm_zho-nld_balanced",
    "BeetleLM/beetlelm_zho-bul_balanced",
    # Simultaneous
    "BeetleLM/beetlelm_zho_L1-eng_L2_simultaneous",
    # Sequential
    "BeetleLM/beetlelm_zho-fra_sequential",
    # Part-Time
    "BeetleLM/beetlelm_zho-fra_part_time",
    # Late
    "BeetleLM/beetlelm_zho-ukr_late",
    "BeetleLM/beetlelm_zho-fas_late",
    "BeetleLM/beetlelm_zho-ind_late",
    "BeetleLM/beetlelm_zho-deu_late",
    "BeetleLM/beetlelm_zho-fra_late",
    # Heritage
    "BeetleLM/beetlelm_zho-fas_heritage",
    "BeetleLM/beetlelm_zho-deu_heritage",
    "BeetleLM/beetlelm_zho-fra_heritage",
    "BeetleLM/beetlelm_zho-bul_heritage",
]

FRENCH_MODELS = [
    "BeetleLM/beetlelm_fra_mono",
    # Balanced
    "BeetleLM/beetlelm_zho-fra_balanced",
    "BeetleLM/beetlelm_nld-fra_balanced",
    "BeetleLM/beetlelm_fra-ind_balanced",
    "BeetleLM/beetlelm_fra-deu_balanced",
    "BeetleLM/beetlelm_fra-ukr_balanced",
    "BeetleLM/beetlelm_fas-fra_balanced",
    "BeetleLM/beetlelm_eng-fra_balanced",
    "BeetleLM/beetlelm_bul-fra_balanced",
    # Simultaneous
    "BeetleLM/beetlelm_bul-fra_simultaneous",
    # Sequential
    "BeetleLM/beetlelm_fra-ukr_sequential",
    "BeetleLM/beetlelm_bul-fra_sequential",
    # Part-Time
    "BeetleLM/beetlelm_zho-fra_part_time",
    "BeetleLM/beetlelm_fra-ukr_part_time",
    "BeetleLM/beetlelm_eng-fra_part_time",
    "BeetleLM/beetlelm_bul-fra_part_time",
    # Late
    "BeetleLM/beetlelm_zho-fra_late",
    "BeetleLM/beetlelm_fra-ind_late",
    "BeetleLM/beetlelm_fra-ukr_late",
    "BeetleLM/beetlelm_bul-fra_late",
    # Heritage
    "BeetleLM/beetlelm_nld-fra_heritage",
    "BeetleLM/beetlelm_zho-fra_heritage",
    "BeetleLM/beetlelm_fra-ukr_heritage",
    "BeetleLM/beetlelm_fra-ind_heritage",
    "BeetleLM/beetlelm_eng-fra_heritage",
    "BeetleLM/beetlelm_fra-deu_heritage",
    "BeetleLM/beetlelm_fas-fra_heritage",
    "BeetleLM/beetlelm_bul-fra_heritage",
]

PERSIAN_MODELS = [
    "BeetleLM/beetlelm_fas_mono",
    # Balanced
    "BeetleLM/beetlelm_zho-fas_balanced",
    "BeetleLM/beetlelm_fas-ukr_balanced",
    "BeetleLM/beetlelm_fas-eng_balanced",
    "BeetleLM/beetlelm_fas-nld_balanced",
    "BeetleLM/beetlelm_fas-ind_balanced",
    "BeetleLM/beetlelm_fas-bul_balanced",
    "BeetleLM/beetlelm_fas-fra_balanced",
    "BeetleLM/beetlelm_fas-deu_balanced",
    # Simultaneous
    "BeetleLM/beetlelm_fas-deu_simultaneous",
    "BeetleLM/beetlelm_fas-nld_simultaneous",
    # Sequential
    "BeetleLM/beetlelm_fas-nld_sequential",
    "BeetleLM/beetlelm_fas-deu_sequential",
    # Part-Time
    "BeetleLM/beetlelm_fas-ind_part_time",
    "BeetleLM/beetlelm_fas-bul_part_time",
    "BeetleLM/beetlelm_fas-eng_part_time",
    "BeetleLM/beetlelm_fas-nld_part_time",
    "BeetleLM/beetlelm_fas-deu_part_time",
    # Late
    "BeetleLM/beetlelm_zho-fas_late",
    "BeetleLM/beetlelm_fas-ind_late",
    "BeetleLM/beetlelm_fas-eng_late",
    "BeetleLM/beetlelm_fas-ukr_late",
    "BeetleLM/beetlelm_fas-nld_late",
    "BeetleLM/beetlelm_fas-deu_late",
    # Heritage
    "BeetleLM/beetlelm_zho-fas_heritage",
    "BeetleLM/beetlelm_fas-ukr_heritage",
    "BeetleLM/beetlelm_fas-ind_heritage",
    "BeetleLM/beetlelm_fas-bul_heritage",
    "BeetleLM/beetlelm_fas-fra_heritage",
    "BeetleLM/beetlelm_fas-nld_heritage",
    "BeetleLM/beetlelm_fas-deu_heritage",
]

BULGARIAN_MODELS = [
    "BeetleLM/beetlelm_bul_mono",
    # Balanced
    "BeetleLM/beetlelm_zho-bul_balanced",
    "BeetleLM/beetlelm_nld-bul_balanced",
    "BeetleLM/beetlelm_bul-ukr_balanced",
    "BeetleLM/beetlelm_bul-ind_balanced",
    "BeetleLM/beetlelm_bul-fra_balanced",
    "BeetleLM/beetlelm_bul-deu_balanced",
    # Simultaneous
    "BeetleLM/beetlelm_bul-fra_simultaneous",
    "BeetleLM/beetlelm_eng-bul_simultaneous",
    "BeetleLM/beetlelm_bul-deu_simultaneous",
    # Sequential
    "BeetleLM/beetlelm_bul-ukr_sequential",
    "BeetleLM/beetlelm_bul-fra_sequential",
    "BeetleLM/beetlelm_bul-deu_sequential",
    # Part-Time
    "BeetleLM/beetlelm_fas-bul_part_time",
    "BeetleLM/beetlelm_bul-ukr_part_time",
    "BeetleLM/beetlelm_bul-fra_part_time",
    "BeetleLM/beetlelm_eng-bul_part_time",
    "BeetleLM/beetlelm_bul-deu_part_time",
    # Late
    "BeetleLM/beetlelm_nld-bul_late",
    "BeetleLM/beetlelm_zho-bul_late",
    "BeetleLM/beetlelm_bul-ukr_late",
    "BeetleLM/beetlelm_bul-fra_late",
    "BeetleLM/beetlelm_bul-deu_late",
    # Heritage
    "BeetleLM/beetlelm_zho-bul_heritage",
    "BeetleLM/beetlelm_nld-bul_heritage",
    "BeetleLM/beetlelm_fas-bul_heritage",
    "BeetleLM/beetlelm_eng-bul_heritage",
    "BeetleLM/beetlelm_bul-ukr_heritage",
    "BeetleLM/beetlelm_bul-fra_heritage",
    "BeetleLM/beetlelm_bul-ind_heritage",
]

# English: all models that contain English in the language pair.
ENGLISH_MODELS = [
    # Balanced
    "BeetleLM/beetlelm_eng-nld_balanced",
    "BeetleLM/beetlelm_nld_L1-eng_L2_balanced",
    "BeetleLM/beetlelm_eng_L1-deu_L2_balanced",
    "BeetleLM/beetlelm_zho_L1-eng_L2_balanced",
    "BeetleLM/beetlelm_zho-eng_balanced",
    "BeetleLM/beetlelm_fas-eng_balanced",
    # Simultaneous
    "BeetleLM/beetlelm_eng-nld_simultaneous",
    "BeetleLM/beetlelm_nld_L1-eng_L2_simultaneous",
    "BeetleLM/beetlelm_deu_L1-eng_L2_simultaneous",
    "BeetleLM/beetlelm_eng_L1-deu_L2_simultaneous",
    "BeetleLM/beetlelm_eng-deu_simultaneous",
    "BeetleLM/beetlelm_zho_L1-eng_L2_simultaneous",
    "BeetleLM/beetlelm_eng-bul_simultaneous",
    # Part-Time
    "BeetleLM/beetlelm_eng-nld_part_time",
    "BeetleLM/beetlelm_fas-eng_part_time",
    "BeetleLM/beetlelm_eng-fra_part_time",
    "BeetleLM/beetlelm_eng-bul_part_time",
    # Late
    "BeetleLM/beetlelm_eng-nld_late",
    "BeetleLM/beetlelm_fas-eng_late",
    # Heritage
    "BeetleLM/beetlelm_eng-deu_heritage",
    "BeetleLM/beetlelm_eng-fra_heritage",
    "BeetleLM/beetlelm_eng-bul_heritage",
]

# ─────────────────────────────────────────────────────────────────────────────
# TRILINGUAL MODELS
# All trilingual models involve the eng / nld / zho triad and are evaluated
# on the blimp_eng, blimp_nl, zhoblimp, xcomps (zho), and xnli (en, zh)
# benchmarks.  They live under the "tri" group key.
# ─────────────────────────────────────────────────────────────────────────────

# Balanced and Heritage trilinguals
TRILINGUAL_BALANCED_MODELS = [
    "BeetleLM/beetlelm_eng_L1-zho_L2-nld_L3_balanced",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_balanced",
    "BeetleLM/beetlelm_nld_L1-eng_L2-zho_L3_balanced",
    "BeetleLM/beetlelm_nld_L1-zho_L2-eng_L3_balanced",
    "BeetleLM/beetlelm_zho_L1-nld_L2-eng_L3_balanced",
    "BeetleLM/beetlelm_zho_L1-eng_L2-nld_L3_balanced",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_heritage",
    "BeetleLM/beetlelm_eng_L1-zho_L2-nld_L3_heritage",
    "BeetleLM/beetlelm_nld_L1-eng_L2-zho_L3_heritage",
    "BeetleLM/beetlelm_nld_L1-zho_L2-eng_L3_heritage",
    "BeetleLM/beetlelm_zho_L1-eng_L2-nld_L3_heritage",
    "BeetleLM/beetlelm_zho_L1-nld_L2-eng_L3_heritage",
]

# Sequential variants: MAML, EWC, combined, and baseline
TRILINGUAL_SEQUENTIAL_MODELS = [
    # EWC + MAML combined
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l5_maml_h25",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l5_maml_h50",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l5_maml_h100",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l20_maml_h25",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l20_maml_h50",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l20_maml_h75",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l20_maml_h100",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l50_maml_h25",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l50_maml_h50",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l50_maml_h75",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l50_maml_h100",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l150_maml_h25",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l150_maml_h50",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l150_maml_h75",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l150_maml_h100",
    # EWC only
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l5",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l20",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l50",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l150",
    # MAML only
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_maml_h25",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_maml_h50",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_maml_h75",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_maml_h100",
    # Baseline
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_baseline",
]

TRILINGUAL_MODELS = TRILINGUAL_BALANCED_MODELS + TRILINGUAL_SEQUENTIAL_MODELS

# ─────────────────────────────────────────────────────────────────────────────
# GROUP REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

MODEL_GROUPS = {
    "nld": DUTCH_MODELS,
    "deu": GERMAN_MODELS,
    "zho": CHINESE_MODELS,
    "fra": FRENCH_MODELS,
    "fas": PERSIAN_MODELS,
    "bul": BULGARIAN_MODELS,
    "eng": ENGLISH_MODELS,
    # Trilingual eng–nld–zho triad.  Evaluated on blimp_eng, blimp_nl,
    # zhoblimp, xcomps (zho), and xnli (en, zh).
    "tri": TRILINGUAL_MODELS,
}

# Flat deduplicated lists — stable order, used for rank::world_size slicing.
ALL_MODELS = list(dict.fromkeys(
    m for group in MODEL_GROUPS.values() for m in group
))

ALL_TRILINGUAL_MODELS = list(dict.fromkeys(TRILINGUAL_MODELS))


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_bilingual_type(repo: str) -> str:
    """
    Return a canonical type string for a repo.

    Trilingual sequential variants carry compound suffixes such as
    'sequential_ewc_l50_maml_h75' or 'sequential_baseline'.  We check the
    most-specific patterns first so that combined EWC+MAML repos are not
    mis-labelled as plain EWC.
    """
    name = repo.split("/")[-1]

    # ── Trilingual sequential sub-types (check before plain "sequential") ──
    if "sequential_ewc" in name and "maml" in name:
        return "sequential_ewc_maml"
    if "sequential_ewc" in name:
        return "sequential_ewc"
    if "sequential_maml" in name:
        return "sequential_maml"
    if "sequential_baseline" in name:
        return "sequential_baseline"

    # ── Standard bilingual types ───────────────────────────────────────────
    for tag in ("mono", "balanced", "simultaneous", "sequential",
                "part_time", "late", "heritage"):
        if tag in name:
            return tag

    return "unknown"


def get_lang_pair(repo: str) -> str:
    """
    Extract the language-pair identifier from a repo name.

    Uses a regex anchor on the first known type suffix so that compound
    trilingual sequential suffixes (e.g. '_sequential_ewc_l5_maml_h25')
    are all stripped correctly, leaving 'eng_L1-nld_L2-zho_L3'.
    """
    name = repo.split("/")[-1].replace("beetlelm_", "")
    m = re.match(
        r"^(.+?)_(mono|balanced|simultaneous|sequential|part_time|late|heritage)\b",
        name,
    )
    return m.group(1) if m else name
