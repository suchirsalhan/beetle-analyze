"""
models.py — Central registry of all BeetleLM model repos, grouped by target language.
Import MODEL_GROUPS to get the full dict, or individual lists by name.
"""

DUTCH_MODELS = [
    # Monolingual
    "BeetleLM/beetlelm_nld_mono",
    # Balanced bilingual
    "BeetleLM/beetlelm_eng-nld_balanced",
    "BeetleLM/beetlelm_nld_L1-eng_L2_balanced",
    "BeetleLM/beetlelm_nld-ukr_balanced",
    "BeetleLM/beetlelm_nld-fra_balanced",
    "BeetleLM/beetlelm_nld-deu_balanced",
    "BeetleLM/beetlelm_nld-ind_balanced",
    "BeetleLM/beetlelm_nld-bul_balanced",
    "BeetleLM/beetlelm_zho-nld_balanced",
    # Simultaneous bilingual
    "BeetleLM/beetlelm_eng-nld_simultaneous",
    "BeetleLM/beetlelm_nld_L1-eng_L2_simultaneous",
    "BeetleLM/beetlelm_fas-nld_simultaneous",
    # Sequential bilingual
    "BeetleLM/beetlelm_fas-nld_sequential",
    # Part-Time bilingual
    "BeetleLM/beetlelm_eng-nld_part_time",
    "BeetleLM/beetlelm_fas-nld_part_time",
    "BeetleLM/beetlelm_nld-deu_part_time",
    # Late bilingual
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
    # Monolingual
    "BeetleLM/beetlelm_deu_mono",
    # Balanced bilingual
    "BeetleLM/beetlelm_deu_L1-eng_L2_balanced",
    "BeetleLM/beetlelm_eng_L1-deu_L2_balanced",
    # Simultaneous bilingual
    "BeetleLM/beetlelm_deu_L1-eng_L2_simultaneous",
    "BeetleLM/beetlelm_eng_L1-deu_L2_simultaneous",
    "BeetleLM/beetlelm_eng-deu_simultaneous",
    "BeetleLM/beetlelm_deu-ukr_simultaneous",
    "BeetleLM/beetlelm_bul-deu_simultaneous",
    # Sequential bilingual
    "BeetleLM/beetlelm_deu-ukr_sequential",
    "BeetleLM/beetlelm_bul-deu_sequential",
    # Part-Time bilingual
    "BeetleLM/beetlelm_nld-deu_part_time",
    "BeetleLM/beetlelm_deu-ukr_part_time",
    "BeetleLM/beetlelm_bul-deu_part_time",
    # Late bilingual
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
    # Monolingual
    "BeetleLM/beetlelm_zho_mono",
    # Balanced bilingual
    "BeetleLM/beetlelm_zho_L1-eng_L2_balanced",
    "BeetleLM/beetlelm_zho-eng_balanced",
    "BeetleLM/beetlelm_zho-ukr_balanced",
    "BeetleLM/beetlelm_zho-fra_balanced",
    "BeetleLM/beetlelm_zho-fas_balanced",
    "BeetleLM/beetlelm_zho-ind_balanced",
    "BeetleLM/beetlelm_zho-deu_balanced",
    "BeetleLM/beetlelm_zho-nld_balanced",
    "BeetleLM/beetlelm_zho-bul_balanced",
    # Simultaneous bilingual
    "BeetleLM/beetlelm_zho_L1-eng_L2_simultaneous",
    # Sequential bilingual
    "BeetleLM/beetlelm_zho-fra_sequential",
    # Part-Time bilingual
    "BeetleLM/beetlelm_zho-fra_part_time",
    # Late bilingual
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
    # Monolingual
    "BeetleLM/beetlelm_fra_mono",
    # Balanced bilingual
    "BeetleLM/beetlelm_zho-fra_balanced",
    "BeetleLM/beetlelm_nld-fra_balanced",
    "BeetleLM/beetlelm_fra-ind_balanced",
    "BeetleLM/beetlelm_fra-deu_balanced",
    "BeetleLM/beetlelm_fra-ukr_balanced",
    "BeetleLM/beetlelm_fas-fra_balanced",
    "BeetleLM/beetlelm_eng-fra_balanced",
    "BeetleLM/beetlelm_bul-fra_balanced",
    # Simultaneous bilingual
    "BeetleLM/beetlelm_bul-fra_simultaneous",
    # Sequential bilingual
    "BeetleLM/beetlelm_fra-ukr_sequential",
    "BeetleLM/beetlelm_bul-fra_sequential",
    # Part-Time bilingual
    "BeetleLM/beetlelm_zho-fra_part_time",
    "BeetleLM/beetlelm_fra-ukr_part_time",
    "BeetleLM/beetlelm_eng-fra_part_time",
    "BeetleLM/beetlelm_bul-fra_part_time",
    # Late bilingual
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
    # Monolingual
    "BeetleLM/beetlelm_fas_mono",
    # Balanced bilingual
    "BeetleLM/beetlelm_zho-fas_balanced",
    "BeetleLM/beetlelm_fas-ukr_balanced",
    "BeetleLM/beetlelm_fas-eng_balanced",
    "BeetleLM/beetlelm_fas-nld_balanced",
    "BeetleLM/beetlelm_fas-ind_balanced",
    "BeetleLM/beetlelm_fas-bul_balanced",
    "BeetleLM/beetlelm_fas-fra_balanced",
    "BeetleLM/beetlelm_fas-deu_balanced",
    # Simultaneous bilingual
    "BeetleLM/beetlelm_fas-deu_simultaneous",
    "BeetleLM/beetlelm_fas-nld_simultaneous",
    # Sequential bilingual
    "BeetleLM/beetlelm_fas-nld_sequential",
    "BeetleLM/beetlelm_fas-deu_sequential",
    # Part-Time bilingual
    "BeetleLM/beetlelm_fas-ind_part_time",
    "BeetleLM/beetlelm_fas-bul_part_time",
    "BeetleLM/beetlelm_fas-eng_part_time",
    "BeetleLM/beetlelm_fas-nld_part_time",
    "BeetleLM/beetlelm_fas-deu_part_time",
    # Late bilingual
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
    # Monolingual
    "BeetleLM/beetlelm_bul_mono",
    # Balanced bilingual
    "BeetleLM/beetlelm_zho-bul_balanced",
    "BeetleLM/beetlelm_nld-bul_balanced",
    "BeetleLM/beetlelm_bul-ukr_balanced",
    "BeetleLM/beetlelm_bul-ind_balanced",
    "BeetleLM/beetlelm_bul-fra_balanced",
    "BeetleLM/beetlelm_bul-deu_balanced",
    # Simultaneous bilingual
    "BeetleLM/beetlelm_bul-fra_simultaneous",
    "BeetleLM/beetlelm_eng-bul_simultaneous",
    "BeetleLM/beetlelm_bul-deu_simultaneous",
    # Sequential bilingual
    "BeetleLM/beetlelm_bul-ukr_sequential",
    "BeetleLM/beetlelm_bul-fra_sequential",
    "BeetleLM/beetlelm_bul-deu_sequential",
    # Part-Time bilingual
    "BeetleLM/beetlelm_fas-bul_part_time",
    "BeetleLM/beetlelm_bul-ukr_part_time",
    "BeetleLM/beetlelm_bul-fra_part_time",
    "BeetleLM/beetlelm_eng-bul_part_time",
    "BeetleLM/beetlelm_bul-deu_part_time",
    # Late bilingual
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

# All 6 groups in one dict
MODEL_GROUPS = {
    "nld": DUTCH_MODELS,
    "deu": GERMAN_MODELS,
    "zho": CHINESE_MODELS,
    "fra": FRENCH_MODELS,
    "fas": PERSIAN_MODELS,
    "bul": BULGARIAN_MODELS,
}

# Flat deduplicated list of every unique model (for GPU splitting)
ALL_MODELS = list(dict.fromkeys(
    m for group in MODEL_GROUPS.values() for m in group
))

def get_bilingual_type(repo: str) -> str:
    """Parse the bilingual condition from a repo name."""
    for tag in ("mono", "balanced", "simultaneous", "sequential",
                "part_time", "late", "heritage"):
        if tag in repo:
            return tag
    return "unknown"

def get_lang_pair(repo: str) -> str:
    """Extract the lang-pair slug from a repo name, e.g. 'eng-nld'."""
    name = repo.split("/")[-1].replace("beetlelm_", "")
    for tag in ("_mono", "_balanced", "_simultaneous", "_sequential",
                "_part_time", "_late", "_heritage"):
        name = name.replace(tag, "")
    return name
