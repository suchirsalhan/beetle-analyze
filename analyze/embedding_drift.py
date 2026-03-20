"""
embedding_drift.py — Embedding space analysis across BeetleLM conditions.

Since only final checkpoints are available (no training trajectory), we
treat the *space of conditions* as a proxy for training history:
  mono → bilingual conditions ordered by assumed L1 exposure intensity

Two analyses
------------
1. VOCABULARY OVERLAP
   For each bilingual model, measure how much of its token-level embedding
   space has shifted relative to the mono baseline.
   Metric: mean cosine distance of shared-vocab embeddings.

2. REPRESENTATIONAL GEOMETRY (condition comparison)
   Project all model embeddings into a shared space (via CKA or PCA) and
   visualise how the representations cluster by condition type, language
   pair, and typological distance.

Outputs
-------
results/embeddings/
  vocab_overlap_<iso>.csv        — per-model vocab overlap stats
  embedding_drift_<iso>.csv      — per-token cosine distance to mono
  cka_matrix_<iso>.csv           — pairwise CKA between all models for lang
  pca_embeddings_<iso>.npz       — 2D PCA coordinates for probe words
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from models import MODEL_GROUPS, get_bilingual_type
from utils import load_model_and_tokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("results/embeddings")

# Probe words used for per-word embedding drift tracking.
# These are high-frequency, semantically stable words per language.
PROBE_WORDS: dict[str, list[str]] = {
    "nld": ["de", "het", "een", "van", "en", "in", "is", "dat", "op", "te"],
    "deu": ["die", "der", "und", "in", "den", "ist", "das", "des", "mit", "zu"],
    "zho": ["的", "是", "在", "了", "不", "有", "我", "他", "这", "人"],
    "fra": ["le", "de", "et", "à", "les", "des", "en", "un", "du", "que"],
    "fas": ["و", "در", "به", "از", "که", "را", "با", "این", "است", "بود"],
    "bul": ["на", "и", "в", "да", "се", "е", "не", "с", "за", "от"],
    "eng": ["the", "of", "and", "to", "a", "in", "is", "it", "that", "was"],
}


# ---------------------------------------------------------------------------
# Helpers: extract embeddings
# ---------------------------------------------------------------------------

def get_input_embeddings(repo: str) -> tuple[torch.Tensor, list[str]]:
    """
    Return (embedding_matrix, vocab_tokens) for *repo*.
    embedding_matrix: shape [vocab_size, hidden_dim], float32 on CPU.
    """
    model, tokenizer = load_model_and_tokenizer(repo)
    emb = model.get_input_embeddings().weight.detach().float().cpu()
    vocab = tokenizer.convert_ids_to_tokens(range(emb.size(0)))
    return emb, vocab


def token_ids_for_words(words: list[str], tokenizer) -> dict[str, list[int]]:
    """
    Map each surface word to its tokenizer token-id(s).
    Returns only single-token words (multi-piece excluded for purity).
    """
    result: dict[str, list[int]] = {}
    for word in words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 1:
            result[word] = ids
    return result


# ---------------------------------------------------------------------------
# 1. Vocabulary overlap & embedding drift
# ---------------------------------------------------------------------------

def compute_vocab_overlap(
    iso_code: str,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    For each bilingual model in MODEL_GROUPS[iso_code], compute:
      - shared_vocab_pct    : fraction of tokens shared with mono tokenizer
      - mean_cosine_dist    : mean cosine distance of shared-vocab embeddings
                              from the mono baseline
      - probe_word_dists    : per-probe-word cosine distance

    Returns DataFrame with one row per bilingual model.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"vocab_overlap_{iso_code}.csv"

    if out_path.exists() and not overwrite:
        print(f"[embed_drift] Loading cached {out_path}")
        return pd.read_csv(out_path)

    mono_repo = f"BeetleLM/beetlelm_{iso_code}_mono"
    print(f"[embed_drift] Loading mono baseline: {mono_repo}")
    mono_emb, mono_vocab = get_input_embeddings(mono_repo)
    mono_vocab_set = {t: i for i, t in enumerate(mono_vocab)}

    _, mono_tokenizer = load_model_and_tokenizer(mono_repo)
    probe_words = PROBE_WORDS.get(iso_code, [])

    candidates = [m for m in MODEL_GROUPS[iso_code] if "mono" not in m]
    rows: list[dict] = []

    for repo in candidates:
        print(f"[embed_drift] Comparing {repo} …")
        bi_emb, bi_vocab = get_input_embeddings(repo)
        _, bi_tokenizer = load_model_and_tokenizer(repo)

        # Shared vocabulary
        bi_vocab_set = {t: i for i, t in enumerate(bi_vocab)}
        shared_tokens = set(mono_vocab_set.keys()) & set(bi_vocab_set.keys())
        shared_pct = len(shared_tokens) / len(mono_vocab_set) * 100

        # Cosine distance for shared tokens
        mono_ids = [mono_vocab_set[t] for t in shared_tokens]
        bi_ids   = [bi_vocab_set[t]   for t in shared_tokens]

        mono_vecs = mono_emb[mono_ids]    # [N, D]
        bi_vecs   = bi_emb[bi_ids]        # [N, D]

        cos_sims = cosine_similarity(mono_vecs, bi_vecs, dim=1).numpy()
        cos_dists = 1.0 - cos_sims

        row = {
            "iso_code":         iso_code,
            "repo":             repo,
            "bilingual_type":   get_bilingual_type(repo),
            "shared_vocab_pct": round(shared_pct, 2),
            "n_shared":         len(shared_tokens),
            "mean_cosine_dist": float(np.mean(cos_dists)),
            "std_cosine_dist":  float(np.std(cos_dists)),
            "median_cosine_dist": float(np.median(cos_dists)),
        }

        # Per-probe-word distances
        probe_map = token_ids_for_words(probe_words, mono_tokenizer)
        for word, (mono_id,) in {k: v for k, v in probe_map.items() if len(v) == 1}.items():
            bi_probe_ids = token_ids_for_words([word], bi_tokenizer)
            if word in bi_probe_ids and len(bi_probe_ids[word]) == 1:
                bi_id = bi_probe_ids[word][0]
                sim = cosine_similarity(
                    mono_emb[mono_id].unsqueeze(0),
                    bi_emb[bi_id].unsqueeze(0),
                ).item()
                row[f"probe_{word}_cosine_dist"] = round(1.0 - sim, 4)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  → saved {out_path}")
    return df


# ---------------------------------------------------------------------------
# 2. CKA — Centred Kernel Alignment between all models for a language
# ---------------------------------------------------------------------------

def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute linear CKA between representation matrices X and Y.
    Both should be [n_samples, d] (using shared probe word embeddings).
    """
    def _centre(K: np.ndarray) -> np.ndarray:
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    K_X = X @ X.T
    K_Y = Y @ Y.T
    numerator   = np.sum(_centre(K_X) * _centre(K_Y))
    denominator = np.linalg.norm(_centre(K_X), "fro") * np.linalg.norm(_centre(K_Y), "fro")
    return float(numerator / (denominator + 1e-10))


def compute_cka_matrix(
    iso_code: str,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Compute pairwise linear CKA between all models in MODEL_GROUPS[iso_code]
    using the shared probe-word embeddings as the representation sample.

    Returns a square DataFrame (repo × repo) of CKA values.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"cka_matrix_{iso_code}.csv"

    if out_path.exists() and not overwrite:
        print(f"[embed_drift] Loading cached {out_path}")
        return pd.read_csv(out_path, index_col=0)

    repos = MODEL_GROUPS[iso_code]
    probe_words = PROBE_WORDS.get(iso_code, [])

    # Collect embeddings for probe words across all models
    # We need consistent probe token IDs — use intersection across all tokenizers
    print("[embed_drift] Collecting shared probe tokens across all models …")

    # First pass: find probe words tokenised to single token in ALL models
    probe_in_all: set[str] | None = None
    tokenizers = {}
    for repo in repos:
        _, tok = load_model_and_tokenizer(repo)
        tokenizers[repo] = tok
        probe_map = token_ids_for_words(probe_words, tok)
        single_tok_words = set(probe_map.keys())
        probe_in_all = single_tok_words if probe_in_all is None else probe_in_all & single_tok_words

    if not probe_in_all:
        print("[embed_drift] WARNING: No probe words are single-token in all models. Skipping CKA.")
        return pd.DataFrame()

    probe_in_all_list = sorted(probe_in_all)
    print(f"  Using {len(probe_in_all_list)} shared probe words: {probe_in_all_list}")

    # Second pass: extract probe vectors for each model
    model_reps: dict[str, np.ndarray] = {}
    for repo in repos:
        emb, vocab = get_input_embeddings(repo)
        tok = tokenizers[repo]
        probe_map = token_ids_for_words(probe_in_all_list, tok)
        vecs = np.stack([emb[probe_map[w][0]].numpy() for w in probe_in_all_list])
        model_reps[repo] = vecs   # [n_probes, hidden_dim]

    # Pairwise CKA
    cka_values = np.zeros((len(repos), len(repos)))
    for i, r1 in enumerate(repos):
        for j, r2 in enumerate(repos):
            cka_values[i, j] = _linear_cka(model_reps[r1], model_reps[r2])

    cka_df = pd.DataFrame(cka_values, index=repos, columns=repos)
    cka_df.to_csv(out_path)
    print(f"  → saved {out_path}")
    return cka_df


# ---------------------------------------------------------------------------
# 3. PCA of probe-word embeddings (for plotting drift across conditions)
# ---------------------------------------------------------------------------

def compute_probe_pca(
    iso_code: str,
    n_components: int = 2,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Project probe-word embeddings from all models into a shared 2D PCA space.

    The PCA is fit on the MONO model embeddings only, then other models are
    projected into the same space — so drift is relative to mono.

    Returns DataFrame: [repo, bilingual_type, word, pc1, pc2]
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"probe_pca_{iso_code}.csv"

    if out_path.exists() and not overwrite:
        print(f"[embed_drift] Loading cached {out_path}")
        return pd.read_csv(out_path)

    probe_words = PROBE_WORDS.get(iso_code, [])
    repos = MODEL_GROUPS[iso_code]
    mono_repo = f"BeetleLM/beetlelm_{iso_code}_mono"

    # Fit PCA on mono embeddings (probe words only)
    mono_emb, _ = get_input_embeddings(mono_repo)
    _, mono_tok = load_model_and_tokenizer(mono_repo)
    mono_probe_map = token_ids_for_words(probe_words, mono_tok)

    available_words = sorted(mono_probe_map.keys())
    mono_vecs = np.stack([mono_emb[mono_probe_map[w][0]].numpy() for w in available_words])

    pca = PCA(n_components=n_components)
    pca.fit(mono_vecs)
    print(f"[embed_drift] PCA variance explained: {pca.explained_variance_ratio_}")

    rows: list[dict] = []

    for repo in repos:
        emb, _ = get_input_embeddings(repo)
        _, tok = load_model_and_tokenizer(repo)
        probe_map = token_ids_for_words(available_words, tok)

        for word in available_words:
            if word not in probe_map:
                continue
            vec = emb[probe_map[word][0]].numpy().reshape(1, -1)
            coords = pca.transform(vec)[0]
            rows.append({
                "repo":           repo,
                "iso_code":       iso_code,
                "bilingual_type": get_bilingual_type(repo),
                "word":           word,
                "pc1":            float(coords[0]),
                "pc2":            float(coords[1]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  → saved {out_path}")
    return df


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

def run_embedding_analysis(
    lang_codes: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, dict]:
    """Run vocab overlap, CKA, and PCA drift for all (or specified) languages."""
    if lang_codes is None:
        lang_codes = list(MODEL_GROUPS.keys())

    results = {}
    for code in lang_codes:
        print(f"\n{'='*60}")
        print(f" Embedding analysis: {code}")
        print(f"{'='*60}")
        results[code] = {
            "vocab_overlap": compute_vocab_overlap(code, overwrite),
            "cka_matrix":    compute_cka_matrix(code, overwrite),
            "probe_pca":     compute_probe_pca(code, overwrite),
        }

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run embedding drift analysis.")
    parser.add_argument("--langs", nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_embedding_analysis(lang_codes=args.langs, overwrite=args.overwrite)
