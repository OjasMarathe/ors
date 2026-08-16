"""
Heaps' law on Japanese anime subtitles — single pooled corpus.

Fits V = K * N^beta over the whole 3.27M-token corpus:
  - document-level shuffling (30 permutations) with a spread band
  - an explicit burn-in cutoff below which the fit is not applied
  - beta refitted over several windows, to show how much the answer
    depends on where the fitting range starts and stops
  - the new-types-per-1000-tokens curve, which is where the slowdown shows
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
# from the headline fit and shaded on the plots.
BURN_IN = 10000

SAMPLE_POINTS = 400

COLOUR = "#2f6f8f"
FIT_COLOUR = "#c1440e"


def load_blocks():
    """Read every show's token stream and cut it into fixed-size blocks."""
    paths = sorted(glob.glob(os.path.join(CORPUS_DIR, "*", "*.txt")))

    if not paths:
        raise SystemExit(f"no corpus files under {CORPUS_DIR}/")

    blocks = []

    for path in paths:
        with open(path, encoding="utf-8") as handle:
            tokens = handle.read().split()

        for start in range(0, len(tokens), BLOCK):
            piece = tokens[start:start + BLOCK]

            if piece:
                blocks.append(piece)

    print(f"loaded {len(paths)} shows -> {len(blocks):,} blocks of <= {BLOCK} tokens")
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


def averaged_curve(blocks):
    """Average V(N) over several shuffles, on a common N grid."""
    curves = []

    for seed in range(PERMUTATIONS):
        curves.append(vocabulary_curve(blocks, seed))

    shortest = min(xs[-1] for xs, _ in curves)
    grid = np.unique(
        np.logspace(0, np.log10(shortest), SAMPLE_POINTS).astype(int)
    ).astype(float)

    stacked = []

    for xs, ys in curves:
        stacked.append(np.interp(grid, xs, ys))

    stacked = np.array(stacked)
    return grid, stacked.mean(axis=0), stacked.std(axis=0)


def fit_window(grid, mean, low, high):
    """Fit log V = log K + beta log N over one range of N."""
    mask = (grid >= low) & (grid <= high)

    if mask.sum() < 5:
        return None

    beta, log_k = np.polyfit(np.log10(grid[mask]), np.log10(mean[mask]), 1)
    k = 10.0 ** log_k

    predicted = log_k + beta * np.log10(grid[mask])
    observed = np.log10(mean[mask])
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return beta, k, r_squared, int(mask.sum())


def plot_loglog(grid, mean, spread, beta, k, path):
    figure, axes = plt.subplots(figsize=(10, 7.5))

    axes.loglog(grid, mean, linewidth=2.0, color=COLOUR,
                label="observed V(N), mean of 30 shuffles")
    axes.fill_between(grid, mean - spread, mean + spread,
                      color=COLOUR, alpha=0.25, linewidth=0,
                      label="+/- 1 s.d. across shuffles")

    fit_range = grid[grid >= BURN_IN]
    fitted = k * fit_range ** beta
    axes.loglog(fit_range, fitted, linestyle="--", linewidth=1.8,
                color=FIT_COLOUR,
                label=f"fit: V = {k:.1f} * N^{beta:.3f}")

    axes.axvspan(1, BURN_IN, color="#cccccc", alpha=0.4, zorder=0)
    axes.text(BURN_IN * 0.8, mean[0] * 1.5,
              f"burn-in\nN < {BURN_IN:,}\nexcluded from fit",
              fontsize=8.5, ha="right", color="#555555")

    axes.set_xlabel("N - tokens seen (log scale)")
    axes.set_ylabel("V - distinct word forms (log scale)")
    axes.set_title("Heaps' law - Japanese anime subtitles, 1983-2024\n"
                   f"{int(grid[-1]):,} tokens, {int(mean[-1]):,} distinct forms",
                   fontsize=12.5, pad=12)
    axes.legend(fontsize=9, loc="upper left")
    axes.grid(True, which="major", linewidth=0.3, alpha=0.35)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_linear(grid, mean, beta, k, path):
    figure, axes = plt.subplots(figsize=(10, 7))

    axes.plot(grid, mean, linewidth=2.0, color=COLOUR, label="observed V(N)")
    axes.fill_between(grid, 0, mean, color=COLOUR, alpha=0.15, linewidth=0)

    fitted = k * grid ** beta
    axes.plot(grid, fitted, linestyle="--", linewidth=1.6, color=FIT_COLOUR,
              label=f"fit: V = {k:.1f} * N^{beta:.3f}")

    for marker in [1_000_000, 2_000_000, 3_000_000]:
        if marker > grid[-1]:
            continue

        value = float(np.interp(marker, grid, mean))
        axes.plot([marker], [value], marker="o", markersize=5, color="#111111")
        axes.annotate(f"{marker / 1e6:.0f}M tokens\n{value:,.0f} types",
                      xy=(marker, value), xytext=(10, -30),
                      textcoords="offset points", fontsize=8.5,
                      color="#333333")

    axes.set_xlabel("N - tokens seen (linear scale)")
    axes.set_ylabel("V - distinct word forms (linear scale)")
    axes.set_title("Vocabulary growth on linear axes\n"
                   "the slope keeps falling but never reaches zero",
                   fontsize=12.5, pad=12)
    axes.legend(fontsize=9, loc="upper left")
    axes.grid(True, linewidth=0.3, alpha=0.3)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_rate(grid, mean, beta, path):
    delta_v = np.diff(mean)
    delta_n = np.diff(grid)
    rate = 1000.0 * delta_v / np.maximum(delta_n, 1)
    midpoints = (grid[:-1] + grid[1:]) / 2.0

    keep = midpoints >= 1000
    reference = midpoints[keep]
    values = rate[keep]

    figure, axes = plt.subplots(figsize=(10, 7))
    axes.loglog(reference, values, linewidth=1.6, color=COLOUR,
                label="new distinct words per 1,000 tokens")

    expected = values[0] * (reference / reference[0]) ** (beta - 1.0)
    axes.loglog(reference, expected, linestyle="--", linewidth=1.5,
                color=FIT_COLOUR,
                label=f"expected decay from beta: slope {beta - 1:.3f}")

    first = values[0]
    last = values[-1]

    axes.annotate(f"{first:.0f} new words\nper 1,000 tokens",
                  xy=(reference[0], first), xytext=(14, -4),
                  textcoords="offset points", fontsize=8.5, color="#333333")
    axes.annotate(f"{last:.0f} new words\nper 1,000 tokens",
                  xy=(reference[-1], last), xytext=(-92, 16),
                  textcoords="offset points", fontsize=8.5, color="#333333")

    axes.set_xlabel("N - tokens seen (log scale)")
    axes.set_ylabel("new distinct words per 1,000 tokens (log scale)")
    axes.set_title("Rate of vocabulary discovery - where the slowdown lives\n"
                   f"a {first / last:.0f}x decline; beta < 1 means exactly this",
                   fontsize=12.5, pad=12)
    axes.legend(fontsize=9)
    axes.grid(True, which="major", linewidth=0.3, alpha=0.35)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    blocks = load_blocks()
    total_tokens = sum(len(block) for block in blocks)

    counts = Counter()

    for block in blocks:
        counts.update(block)

    print(f"corpus: {total_tokens:,} tokens, {len(counts):,} distinct forms")
    print(f"building vocabulary curves ({PERMUTATIONS} shuffles) ...")

    grid, mean, spread = averaged_curve(blocks)

    beta, k, r_squared, points = fit_window(grid, mean, BURN_IN, grid[-1])

    windows = [
        ("1 - all (no burn-in)", 1, grid[-1]),
        ("10,000 - all (headline)", BURN_IN, grid[-1]),
        ("10,000 - 100,000", BURN_IN, 100_000),
        ("100,000 - 1,000,000", 100_000, 1_000_000),
        ("1,000,000 - all", 1_000_000, grid[-1]),
    ]

    rows = []

    print()
    print(f"{'fitting window':<26}{'beta':>9}{'K':>9}{'R2':>10}{'points':>8}")
    print("-" * 62)

    for label, low, high in windows:
        fitted = fit_window(grid, mean, low, high)

        if fitted is None:
            continue

        window_beta, window_k, window_r2, window_points = fitted

        rows.append({
            "window": label,
            "beta": round(window_beta, 4),
            "K": round(window_k, 3),
            "r_squared": round(window_r2, 5),
            "points": window_points,
        })

        print(f"{label:<26}{window_beta:>9.4f}{window_k:>9.2f}"
              f"{window_r2:>10.5f}{window_points:>8}")

    predicted_end = k * grid[-1] ** beta

    print()
    print(f"headline fit:  V = {k:.2f} * N^{beta:.4f}   (R2 = {r_squared:.5f}, "
          f"{points} points)")
    print(f"at N = {int(grid[-1]):,}  predicted V = {predicted_end:,.0f}, "
          f"actual V = {int(mean[-1]):,} "
          f"({100 * (predicted_end / mean[-1] - 1):+.1f}%)")

    with open(os.path.join(OUT_DIR, "heaps_parameters.csv"),
              "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with open(os.path.join(OUT_DIR, "heaps_curve.csv"),
              "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["N", "V_mean", "V_sd"])

        for n, v, s in zip(grid, mean, spread):
            writer.writerow([int(n), round(float(v), 2), round(float(s), 2)])

    plot_loglog(grid, mean, spread, beta, k,
                os.path.join(OUT_DIR, "heaps_loglog.png"))
    plot_linear(grid, mean, beta, k,
                os.path.join(OUT_DIR, "heaps_linear.png"))
    plot_rate(grid, mean, beta,
              os.path.join(OUT_DIR, "heaps_discovery_rate.png"))

    print()
    print("wrote plots, heaps_parameters.csv and heaps_curve.csv to", OUT_DIR)


if __name__ == "__main__":
    main()
