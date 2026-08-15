"""
Heaps' law on Japanese anime subtitles, grouped into 3-year bins.

Fits V = K * N^beta for each bin and for the pooled corpus, with:
  - document-level shuffling (30 permutations) and a spread band
  - an explicit burn-in cutoff below which the fit is not applied
  - the new-types-per-1000-tokens curve, which is where the visible
    slowdown ("plateau") actually shows up
  - a Zipf alpha per bin for the beta = 1/alpha consistency check
"""

import csv
import glob
import os
import random
from collections import Counter

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CORPUS_DIR = "heaps_corpus"
OUT_DIR = "out/heaps-anime"

# Pseudo-documents. Mean real episode length is 477 tokens, so 500-token
# blocks approximate episode granularity while giving many shuffle units.
BLOCK = 500

PERMUTATIONS = 30

# Heaps is not a power law at small N. Everything below this is excluded
# from the fit and shaded on the plots.
BURN_IN = 10000

MIN_TOKENS_FOR_FIT = 100000

SAMPLE_POINTS = 400


def load_blocks(pattern):
    """Read token streams and cut them into fixed-size pseudo-documents."""
    blocks = []

    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as handle:
            tokens = handle.read().split()

        for start in range(0, len(tokens), BLOCK):
            piece = tokens[start:start + BLOCK]

            if piece:
                blocks.append(piece)

    return blocks


def vocabulary_curve(blocks, seed):
    """One shuffled pass: returns sampled (N, V) points."""
    order = list(range(len(blocks)))
    rng = random.Random(seed)
    rng.shuffle(order)

    total = sum(len(blocks[index]) for index in order)
    targets = np.unique(
        np.logspace(0, np.log10(max(total, 2)), SAMPLE_POINTS).astype(int)
    )
    targets = targets[targets >= 1]

    seen = set()
    counted = 0
    pointer = 0
    xs = []
    ys = []

    for index in order:
        for token in blocks[index]:
            seen.add(token)
            counted += 1

            while pointer < len(targets) and counted >= targets[pointer]:
                xs.append(counted)
                ys.append(len(seen))
                pointer += 1

    if not xs or xs[-1] != counted:
        xs.append(counted)
        ys.append(len(seen))

    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def averaged_curve(blocks, permutations=PERMUTATIONS):
    """Average V(N) over several shuffles, on a common N grid."""
    curves = []

    for seed in range(permutations):
        xs, ys = vocabulary_curve(blocks, seed)
        curves.append((xs, ys))

    shortest = min(xs[-1] for xs, _ in curves)
    grid = np.unique(
        np.logspace(0, np.log10(shortest), SAMPLE_POINTS).astype(int)
    ).astype(float)

    stacked = []

    for xs, ys in curves:
        stacked.append(np.interp(grid, xs, ys))

    stacked = np.array(stacked)
    return grid, stacked.mean(axis=0), stacked.std(axis=0)


def fit_heaps(grid, mean, burn_in=BURN_IN):
    """Least squares on log V = log K + beta log N, above the burn-in."""
    mask = grid >= burn_in

    if mask.sum() < 5:
        return None

    beta, log_k = np.polyfit(np.log10(grid[mask]), np.log10(mean[mask]), 1)
    k = 10 ** log_k

    predicted = log_k + beta * np.log10(grid[mask])
    residuals = np.log10(mean[mask]) - predicted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((np.log10(mean[mask]) - np.log10(mean[mask]).mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return beta, k, r_squared


def zipf_alpha(blocks):
    """Rank-frequency exponent, for the beta = 1/alpha consistency check."""
    counts = Counter()

    for block in blocks:
        counts.update(block)

    frequencies = np.array([count for _, count in counts.most_common()],
                           dtype=float)
    ranks = np.arange(1, len(frequencies) + 1, dtype=float)
    slope, _ = np.polyfit(np.log10(ranks), np.log10(frequencies), 1)

    return -slope, len(frequencies)


def new_types_rate(grid, mean):
    """New distinct words per 1,000 tokens, as a function of corpus size."""
    delta_v = np.diff(mean)
    delta_n = np.diff(grid)
    rate = 1000.0 * delta_v / np.maximum(delta_n, 1)
    midpoints = (grid[:-1] + grid[1:]) / 2.0
    return midpoints, rate


def plot_loglog(results, path):
    figure, axes = plt.subplots(figsize=(10, 7.5))
    colours = plt.cm.viridis(np.linspace(0.05, 0.9, len(results)))

    for (name, entry), colour in zip(results.items(), colours):
        grid = entry["grid"]
        mean = entry["mean"]
        spread = entry["spread"]

        axes.loglog(grid, mean, linewidth=1.5, color=colour, label=name)
        axes.fill_between(grid, mean - spread, mean + spread,
                          color=colour, alpha=0.18, linewidth=0)

    axes.axvspan(1, BURN_IN, color="#cccccc", alpha=0.35, zorder=0)
    axes.text(BURN_IN * 0.85, axes.get_ylim()[0] * 1.6,
              f"burn-in\n(N < {BURN_IN:,}, excluded from fit)",
              fontsize=8, ha="right", color="#555555")

    axes.set_xlabel("N — tokens seen (log scale)")
    axes.set_ylabel("V — distinct word forms (log scale)")
    axes.set_title("Heaps' law by 3-year bin — Japanese anime subtitles\n"
                   "mean of 30 document shuffles, band = ±1 s.d.",
                   fontsize=12, pad=12)
    axes.legend(fontsize=8, ncol=2, loc="upper left")
    axes.grid(True, which="major", linewidth=0.3, alpha=0.35)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_linear(results, path):
    figure, axes = plt.subplots(figsize=(10, 7))
    colours = plt.cm.viridis(np.linspace(0.05, 0.9, len(results)))

    for (name, entry), colour in zip(results.items(), colours):
        axes.plot(entry["grid"], entry["mean"], linewidth=1.6,
                  color=colour, label=name)

    axes.set_xlabel("N — tokens seen (linear scale)")
    axes.set_ylabel("V — distinct word forms (linear scale)")
    axes.set_title("Vocabulary growth on linear axes — the slope keeps falling\n"
                   "it never becomes flat, but the curvature is the point",
                   fontsize=12, pad=12)
    axes.legend(fontsize=8, ncol=2)
    axes.grid(True, linewidth=0.3, alpha=0.3)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_rate(results, path):
    figure, axes = plt.subplots(figsize=(10, 7))
    colours = plt.cm.viridis(np.linspace(0.05, 0.9, len(results)))

    for (name, entry), colour in zip(results.items(), colours):
        midpoints, rate = new_types_rate(entry["grid"], entry["mean"])
        keep = midpoints >= 1000
        axes.loglog(midpoints[keep], rate[keep], linewidth=1.5,
                    color=colour, label=name)

    axes.set_xlabel("N — tokens seen (log scale)")
    axes.set_ylabel("new distinct words per 1,000 tokens (log scale)")
    axes.set_title("Rate of vocabulary discovery — where the slowdown lives\n"
                   "a straight decline here is exactly what beta < 1 means",
                   fontsize=12, pad=12)
    axes.legend(fontsize=8, ncol=2)
    axes.grid(True, which="major", linewidth=0.3, alpha=0.35)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_beta_over_time(rows, path):
    usable = [row for row in rows
              if row["beta"] != "" and row["bin"] != "POOLED"]

    if len(usable) < 2:
        return

    labels = [row["bin"] for row in usable]
    betas = [float(row["beta"]) for row in usable]
    ks = [float(row["K"]) for row in usable]

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    positions = np.arange(len(labels))

    top.plot(positions, betas, marker="o", linewidth=1.6, color="#3f6b8c")
    top.set_ylabel("beta")
    top.set_title("Heaps parameters across time bins", fontsize=12, pad=10)
    top.grid(True, linewidth=0.3, alpha=0.35)

    bottom.plot(positions, ks, marker="s", linewidth=1.6, color="#8c5a3f")
    bottom.set_ylabel("K")
    bottom.set_xticks(positions)
    bottom.set_xticklabels(labels, rotation=45, ha="right")
    bottom.grid(True, linewidth=0.3, alpha=0.35)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    bins = sorted(
        directory for directory in os.listdir(CORPUS_DIR)
        if os.path.isdir(os.path.join(CORPUS_DIR, directory))
    )

    loaded = {}

    for name in bins:
        blocks = load_blocks(os.path.join(CORPUS_DIR, name, "*.txt"))
        tokens = sum(len(block) for block in blocks)
        loaded[name] = {"blocks": blocks, "tokens": tokens}

    # Bins below the fitting threshold are pooled into one early group so the
    # 1980s-1990s material is still represented instead of being dropped.
    small = [name for name in bins if loaded[name]["tokens"] < MIN_TOKENS_FOR_FIT]
    large = [name for name in bins if loaded[name]["tokens"] >= MIN_TOKENS_FOR_FIT]

    groups = {}

    # Early bins are individually too thin to fit, so they are merged into one
    # pre-modern group. Any thin bin at the recent end is folded into the last
    # substantial bin rather than left dangling as a one-show outlier.
    early_small = [name for name in small if int(name.split("-")[0]) < 2004]
    late_small = [name for name in small if int(name.split("-")[0]) >= 2004]

    if early_small:
        merged = []

        for name in early_small:
            merged.extend(loaded[name]["blocks"])

        span = f"{early_small[0].split('-')[0]}-{early_small[-1].split('-')[1]}"
        groups[f"{span} (merged)"] = merged

    for index, name in enumerate(large):
        blocks = list(loaded[name]["blocks"])
        label = name

        if index == len(large) - 1 and late_small:
            for extra in late_small:
                blocks.extend(loaded[extra]["blocks"])

            label = f"{name.split('-')[0]}-{late_small[-1].split('-')[1]}"

        groups[label] = blocks

    pooled = []

    for blocks in groups.values():
        pooled.extend(blocks)

    print(f"{'group':<22}{'tokens':>11}{'types':>10}{'beta':>8}"
          f"{'K':>9}{'R2':>8}{'alpha':>8}{'1/alpha':>9}")
    print("-" * 85)

    results = {}
    rows = []

    for name, blocks in list(groups.items()) + [("POOLED", pooled)]:
        grid, mean, spread = averaged_curve(blocks)
        tokens = sum(len(block) for block in blocks)

        fitted = fit_heaps(grid, mean)
        alpha, types = zipf_alpha(blocks)

        if fitted is None:
            beta = k = r_squared = ""
        else:
            beta, k, r_squared = fitted

        if name != "POOLED":
            results[name] = {"grid": grid, "mean": mean, "spread": spread}

        rows.append({
            "bin": name,
            "tokens": tokens,
            "types": types,
            "beta": round(beta, 4) if fitted else "",
            "K": round(k, 2) if fitted else "",
            "r_squared": round(r_squared, 4) if fitted else "",
            "zipf_alpha": round(alpha, 4),
            "one_over_alpha": round(1.0 / alpha, 4),
            "burn_in": BURN_IN,
        })

        print(f"{name:<22}{tokens:>11,}{types:>10,}"
              f"{(f'{beta:.3f}' if fitted else '-'):>8}"
              f"{(f'{k:.1f}' if fitted else '-'):>9}"
              f"{(f'{r_squared:.4f}' if fitted else '-'):>8}"
              f"{alpha:>8.3f}{1.0 / alpha:>9.3f}")

    with open(os.path.join(OUT_DIR, "heaps_parameters.csv"),
              "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grid, mean, spread = averaged_curve(pooled)
    results_with_pooled = dict(results)
    results_with_pooled["POOLED (all bins)"] = {
        "grid": grid, "mean": mean, "spread": spread,
    }

    plot_loglog(results_with_pooled, os.path.join(OUT_DIR, "heaps_loglog.png"))
    plot_linear(results_with_pooled, os.path.join(OUT_DIR, "heaps_linear.png"))
    plot_rate(results_with_pooled, os.path.join(OUT_DIR, "heaps_discovery_rate.png"))
    plot_beta_over_time(rows, os.path.join(OUT_DIR, "heaps_beta_over_time.png"))

    print()
    print("wrote plots and heaps_parameters.csv to", OUT_DIR)


if __name__ == "__main__":
    main()
