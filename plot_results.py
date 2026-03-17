#!/usr/bin/env python3
"""
plot_results.py — Generate publication-quality figures from BeetleLM eval CSVs.

Figures produced (saved to --output_dir as 300 dpi PNGs)
---------------------------------------------------------
  01_learning_curves_{benchmark}.png  — accuracy vs step, one line per model,
                                        faceted by eval language, coloured by
                                        bilingual type
  02_final_heatmap_{benchmark}.png    — model × eval-language heatmap of final
                                        accuracy (main checkpoint)
  03_barplot_by_type_{benchmark}.png  — grouped bar chart: bilingual type × lang
  04_scatter_l1_l2.png                — L1 vs L2 accuracy scatter (all benchmarks)
  05_phenomenon_breakdown.png         — per-phenomenon accuracy for MultiBLiMP
                                        (requires 'phenomenon' column if present)

Usage
-----
  python plot_results.py --results_dir results/ --output_dir figures/

  # Restrict to one benchmark
  python plot_results.py --benchmark multiblimp

  # Restrict to one language group
  python plot_results.py --lang_group deu
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Visual style (paper-friendly) ────────────────────────────────────────────
FONT_SIZE   = 9
LABEL_SIZE  = 8
TITLE_SIZE  = 10
DPI         = 300
FIGW_COL    = 3.5    # single-column width (inches) for typical ACL/EMNLP layout
FIGW_FULL   = 7.0    # double-column / full-width

# Colour palette per bilingual type (colourblind-safe)
TYPE_PALETTE = {
    "mono"        : "#333333",
    "balanced"    : "#1f77b4",
    "simultaneous": "#ff7f0e",
    "sequential"  : "#2ca02c",
    "part_time"   : "#d62728",
    "late"        : "#9467bd",
    "heritage"    : "#8c564b",
    "unknown"     : "#7f7f7f",
}

TYPE_ORDER = ["mono", "balanced", "simultaneous", "sequential",
              "part_time", "late", "heritage"]

matplotlib.rcParams.update({
    "font.family"     : "serif",
    "font.size"       : FONT_SIZE,
    "axes.titlesize"  : TITLE_SIZE,
    "axes.labelsize"  : LABEL_SIZE,
    "xtick.labelsize" : LABEL_SIZE,
    "ytick.labelsize" : LABEL_SIZE,
    "legend.fontsize" : LABEL_SIZE,
    "figure.dpi"      : DPI,
    "savefig.dpi"     : DPI,
    "savefig.bbox"    : "tight",
    "savefig.pad_inches": 0.02,
    "axes.spines.top" : False,
    "axes.spines.right": False,
})


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_results(results_dir: str, benchmark_filter=None,
                     lang_group_filter=None) -> pd.DataFrame:
    dfs = []
    for p in Path(results_dir).glob("*.csv"):
        try:
            df = pd.read_csv(p)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: could not read {p}: {e}")

    if not dfs:
        sys.exit(f"No CSVs found in {results_dir}")

    df = pd.concat(dfs, ignore_index=True).drop_duplicates()

    # Derive step number for sorting / x-axis
    def parse_step(ckpt):
        if isinstance(ckpt, str) and ckpt.startswith("step-"):
            return int(ckpt.split("-")[1])
        return -1   # 'main' → assign as final

    df["step"]   = df["checkpoint"].apply(parse_step)
    df["is_main"]= df["checkpoint"] == "main"

    # Assign max step + 1 to 'main' so it plots last
    max_step = df.loc[df["step"] >= 0, "step"].max() if (df["step"] >= 0).any() else 0
    df.loc[df["is_main"], "step"] = max_step + 1

    # Clean bilingual_type
    df["bilingual_type"] = df["bilingual_type"].fillna("unknown")

    if benchmark_filter:
        df = df[df["benchmark"] == benchmark_filter]
    if lang_group_filter:
        df = df[df["lang_pair"].str.contains(lang_group_filter)]

    return df


def short_name(repo: str) -> str:
    """Strip the HF org prefix and 'beetlelm_' for plot labels."""
    return repo.split("/")[-1].replace("beetlelm_", "")


# ── Figure 1 — Learning curves ────────────────────────────────────────────────

def plot_learning_curves(df: pd.DataFrame, benchmark: str, out_dir: str):
    bdf = df[df["benchmark"] == benchmark].copy()
    if bdf.empty:
        return

    langs = sorted(bdf["eval_language"].unique())
    types = [t for t in TYPE_ORDER if t in bdf["bilingual_type"].unique()]

    ncols = min(len(langs), 3)
    nrows = (len(langs) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(FIGW_FULL, nrows * 2.2),
        sharey=True, sharex=False
    )
    axes = np.array(axes).flatten()

    for ax, lang in zip(axes, langs):
        ldf = bdf[bdf["eval_language"] == lang]
        for bil_type in types:
            tdf = ldf[ldf["bilingual_type"] == bil_type].sort_values("step")
            if tdf.empty:
                continue
            # Aggregate over models of the same type (mean ± std)
            grouped = tdf.groupby("step")["accuracy"]
            means   = grouped.mean()
            stds    = grouped.std().fillna(0)
            steps   = means.index.values

            ax.plot(steps, means.values,
                    color=TYPE_PALETTE[bil_type],
                    label=bil_type.replace("_", "-"),
                    linewidth=1.2, marker="o", markersize=2)
            ax.fill_between(steps,
                            means - stds, means + stds,
                            color=TYPE_PALETTE[bil_type], alpha=0.12)

        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.set_title(lang)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.35, 1.02)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # Hide unused axes
    for ax in axes[len(langs):]:
        ax.set_visible(False)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=len(types),
               bbox_to_anchor=(0.5, -0.02),
               frameon=False, title="Bilingual type")

    fig.suptitle(f"{benchmark.upper()} — Learning curves", y=1.01, fontsize=TITLE_SIZE)
    out = os.path.join(out_dir, f"01_learning_curves_{benchmark}.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 2 — Final accuracy heatmap ────────────────────────────────────────

def plot_heatmap(df: pd.DataFrame, benchmark: str, out_dir: str):
    bdf = df[(df["benchmark"] == benchmark) & (df["is_main"])].copy()
    if bdf.empty:
        # Fall back to last checkpoint per model × lang
        bdf = (df[df["benchmark"] == benchmark]
               .sort_values("step")
               .groupby(["model", "eval_language"])
               .last()
               .reset_index())

    bdf["model_short"] = bdf["model"].apply(short_name)
    pivot = bdf.pivot_table(
        index="model_short", columns="eval_language",
        values="accuracy", aggfunc="mean"
    )

    # Sort rows by bilingual type then lang pair
    type_order_map = {t: i for i, t in enumerate(TYPE_ORDER)}
    row_order = (
        bdf[["model_short", "bilingual_type"]]
        .drop_duplicates()
        .set_index("model_short")["bilingual_type"]
        .map(type_order_map)
    )
    pivot = pivot.loc[pivot.index.intersection(row_order.sort_values().index)]

    fig_h = max(4, len(pivot) * 0.22)
    fig_w = max(FIGW_COL, len(pivot.columns) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        pivot, ax=ax,
        cmap="RdYlGn", vmin=0.5, vmax=1.0,
        linewidths=0.3, linecolor="white",
        annot=True, fmt=".2f", annot_kws={"size": 5},
        cbar_kws={"shrink": 0.6, "label": "Accuracy"}
    )
    ax.set_title(f"{benchmark.upper()} — Final accuracy (main checkpoint)", pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    out = os.path.join(out_dir, f"02_final_heatmap_{benchmark}.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 3 — Bar chart by bilingual type ────────────────────────────────────

def plot_barplot_by_type(df: pd.DataFrame, benchmark: str, out_dir: str):
    bdf = df[(df["benchmark"] == benchmark) & (df["is_main"])].copy()
    if bdf.empty:
        bdf = (df[df["benchmark"] == benchmark]
               .sort_values("step")
               .groupby(["model", "eval_language"])
               .last()
               .reset_index())

    if bdf.empty:
        return

    langs = sorted(bdf["eval_language"].unique())
    types = [t for t in TYPE_ORDER if t in bdf["bilingual_type"].unique()]

    agg = (bdf.groupby(["bilingual_type", "eval_language"])["accuracy"]
              .agg(["mean", "sem"])
              .reset_index())

    ncols = min(len(langs), 3)
    nrows = (len(langs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(FIGW_FULL, nrows * 2.5),
                             sharey=True)
    axes = np.array(axes).flatten()

    for ax, lang in zip(axes, langs):
        ldf = agg[agg["eval_language"] == lang]
        ldf = ldf.set_index("bilingual_type").reindex(types)
        means = ldf["mean"].values
        sems  = ldf["sem"].fillna(0).values
        x     = np.arange(len(types))
        bars  = ax.bar(x, means, yerr=sems, capsize=2,
                       color=[TYPE_PALETTE[t] for t in types],
                       edgecolor="white", linewidth=0.5, width=0.7)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("_", "\n") for t in types],
                           fontsize=6, rotation=0)
        ax.set_title(lang)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.35, 1.05)

    for ax in axes[len(langs):]:
        ax.set_visible(False)

    fig.suptitle(f"{benchmark.upper()} — Accuracy by bilingual type", y=1.01)
    out = os.path.join(out_dir, f"03_barplot_by_type_{benchmark}.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 4 — L1 vs L2 accuracy scatter ─────────────────────────────────────

def plot_l1_l2_scatter(df: pd.DataFrame, out_dir: str):
    """
    For bilingual models trained on two languages, compare accuracy on L1 vs L2.
    We infer L1/L2 from the lang_pair field ('l1-l2' or 'l1_l1-l2_l2').
    """
    # Only 'main' checkpoint, collapse across benchmarks by taking mean
    mdf = df[df["is_main"]].copy()
    if mdf.empty:
        return

    # Models with explicit L1/L2 ordering in repo name
    l1l2_models = mdf[mdf["model"].str.contains("_L1|_L2", regex=True)].copy()
    if l1l2_models.empty:
        print("No L1/L2-labelled models found for scatter plot — skipping.")
        return

    # Map eval_language → language code in model name
    # This requires the eval_language to match one of the two languages in the pair
    # We do a best-effort join: for each model get L1 lang and L2 lang from repo name
    import re
    rows = []
    for repo, grp in l1l2_models.groupby("model"):
        name = repo.split("/")[-1]
        m_l1 = re.search(r"([a-z]{3})_L1", name)
        m_l2 = re.search(r"_L2_([a-z]{3})", name)
        if not (m_l1 and m_l2):
            continue
        l1_code = m_l1.group(1)
        l2_code = m_l2.group(1)

        # Find accuracies for those languages in eval results
        l1_rows = grp[grp["eval_language"].str.lower().str[:2] == l1_code[:2]]
        l2_rows = grp[grp["eval_language"].str.lower().str[:2] == l2_code[:2]]
        if l1_rows.empty or l2_rows.empty:
            continue

        l1_acc = l1_rows["accuracy"].mean()
        l2_acc = l2_rows["accuracy"].mean()
        bil    = grp["bilingual_type"].iloc[0]
        rows.append({"model": short_name(repo), "l1_acc": l1_acc,
                     "l2_acc": l2_acc, "bilingual_type": bil,
                     "l1": l1_code, "l2": l2_code})

    if not rows:
        print("Could not extract L1/L2 pairs — skipping scatter plot.")
        return

    sdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(FIGW_FULL * 0.7, FIGW_FULL * 0.7))

    for bil_type, grp in sdf.groupby("bilingual_type"):
        ax.scatter(grp["l1_acc"], grp["l2_acc"],
                   color=TYPE_PALETTE.get(bil_type, "#999"),
                   label=bil_type.replace("_", "-"),
                   s=30, alpha=0.85, edgecolors="white", linewidths=0.4)

    # Diagonal line (L1 = L2)
    lim = [0.4, 1.0]
    ax.plot(lim, lim, "--", color="grey", linewidth=0.8, alpha=0.6)
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("L1 accuracy")
    ax.set_ylabel("L2 accuracy")
    ax.set_title("L1 vs L2 accuracy (main checkpoint, all benchmarks)")
    ax.legend(frameon=False, title="Type", ncol=2, fontsize=7)

    out = os.path.join(out_dir, "04_scatter_l1_l2.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 5 — Per-phenomenon breakdown (MultiBLiMP) ─────────────────────────

def plot_phenomenon_breakdown(df: pd.DataFrame, out_dir: str):
    """Only runs if a 'phenomenon' column is present in the data."""
    if "phenomenon" not in df.columns:
        print("No 'phenomenon' column — skipping breakdown plot.")
        return

    mdf = df[(df["benchmark"] == "multiblimp") & (df["is_main"])].copy()
    if mdf.empty:
        return

    agg = (mdf.groupby(["phenomenon", "bilingual_type", "eval_language"])
              ["accuracy"].mean().reset_index())

    langs = sorted(agg["eval_language"].unique())
    types = [t for t in TYPE_ORDER if t in agg["bilingual_type"].unique()]

    for lang in langs:
        ldf = agg[agg["eval_language"] == lang]
        phenomena = sorted(ldf["phenomenon"].unique())
        pivot = (ldf.pivot_table(index="phenomenon", columns="bilingual_type",
                                 values="accuracy", aggfunc="mean")
                    .reindex(columns=[t for t in types if t in ldf["bilingual_type"].unique()])
                    .reindex(phenomena))

        fig, ax = plt.subplots(figsize=(FIGW_FULL, max(3, len(phenomena) * 0.35)))
        x = np.arange(len(phenomena))
        width = 0.8 / len(pivot.columns)
        for i, col in enumerate(pivot.columns):
            ax.barh(x + i * width, pivot[col].values,
                    height=width * 0.9,
                    color=TYPE_PALETTE.get(col, "#999"),
                    label=col.replace("_", "-"))
        ax.set_yticks(x + width * (len(pivot.columns) - 1) / 2)
        ax.set_yticklabels(phenomena, fontsize=6)
        ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.set_xlabel("Accuracy")
        ax.set_title(f"MultiBLiMP — Phenomenon breakdown ({lang})")
        ax.legend(frameon=False, ncol=2, fontsize=6)
        ax.set_xlim(0.3, 1.0)

        out = os.path.join(out_dir, f"05_phenomenon_{lang.lower()}.png")
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved: {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results",
                   help="Directory containing *_results.csv files")
    p.add_argument("--output_dir",  default="figures",
                   help="Directory to write PNG figures")
    p.add_argument("--benchmark",   default=None,
                   help="Restrict to one benchmark (e.g. multiblimp)")
    p.add_argument("--lang_group",  default=None,
                   help="Restrict to models containing this lang code (e.g. deu)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_all_results(args.results_dir, args.benchmark, args.lang_group)
    print(f"Loaded {len(df):,} rows from {args.results_dir}")
    print(f"Benchmarks : {sorted(df['benchmark'].unique())}")
    print(f"Languages  : {sorted(df['eval_language'].unique())}")
    print(f"Models     : {df['model'].nunique()}")
    print()

    benchmarks = sorted(df["benchmark"].unique())

    for bm in benchmarks:
        print(f"── {bm} ──────────────────────────────")
        plot_learning_curves(df, bm, args.output_dir)
        plot_heatmap(df, bm, args.output_dir)
        plot_barplot_by_type(df, bm, args.output_dir)

    plot_l1_l2_scatter(df, args.output_dir)
    plot_phenomenon_breakdown(df, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
