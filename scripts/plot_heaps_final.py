"""
Final Heaps figures — observed data only, no fitted power law drawn.

A single fitted line would represent one arbitrary choice of fitting window,
and the local-slope panel shows that no such choice is defensible. So the
figures show what was measured and let the slope decline speak for itself.

Reads out/heaps-anime/heaps_curve.csv (N, V_mean, V_sd) produced by
analyse_heaps.py, so the 30 shuffles are not recomputed.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CURVE_CSV = "out/heaps-anime/heaps_curve.csv"
OUT_DIR = "out/heaps-anime"

WINDOW = 0.35

COLOUR = "#2f6f8f"
SLOPE_COLOUR = "#c1440e"


def load_curve(path):
    grid = []
    mean = []
    spread = []

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            grid.append(float(row["N"]))
            mean.append(float(row["V_mean"]))
            spread.append(float(row["V_sd"]))

    return np.array(grid), np.array(mean), np.array(spread)


def local_slope(grid, mean, window=WINDOW):
    """d(log V)/d(log N) measured over a rolling window in log space."""
    log_n = np.log10(grid)
    log_v = np.log10(mean)

    centres = []
    slopes = []

    for index in range(len(log_n)):
        low = log_n[index] - window
        high = log_n[index] + window
        mask = (log_n >= low) & (log_n <= high)

        if mask.sum() < 6:
            continue

        slope, _ = np.polyfit(log_n[mask], log_v[mask], 1)
        centres.append(grid[index])
        slopes.append(slope)

    return np.array(centres), np.array(slopes)


def plot_main(grid, mean, spread, centres, slopes, path):
    """Vocabulary growth and its local slope, sharing an x-axis."""
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 9.5), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1]},
    )

    top.loglog(grid, mean, linewidth=2.0, color=COLOUR,
               label="observed V(N), mean of 30 shuffles")
    top.fill_between(grid, mean - spread, mean + spread,
                     color=COLOUR, alpha=0.25, linewidth=0,
                     label="+/- 1 s.d. across shuffles")

    top.set_ylabel("V - distinct word forms (log scale)")
    top.set_title("Heaps' law - Japanese anime subtitles, 1983-2024\n"
                  f"{int(grid[-1]):,} tokens, {int(mean[-1]):,} distinct forms",
                  fontsize=12.5, pad=12)
    top.legend(fontsize=9, loc="upper left")
    top.grid(True, which="major", linewidth=0.3, alpha=0.35)

    bottom.semilogx(centres, slopes, linewidth=1.8, color=SLOPE_COLOUR)
    bottom.axhline(1.0, linestyle=":", color="#888888", linewidth=1.2)
    bottom.text(centres[0] * 1.4, 1.01,
                "slope = 1: every token is a new word",
                fontsize=8, color="#666666")

    for target in [1e4, 1e5, 1e6, 3e6]:
        if target > centres[-1]:
            continue

        value = float(np.interp(target, centres, slopes))
        bottom.plot([target], [value], marker="o", markersize=4.5,
                    color="#111111")
        bottom.annotate(f"{value:.2f}", xy=(target, value), xytext=(6, 8),
                        textcoords="offset points", fontsize=8.5,
                        color="#333333")

    bottom.set_xlabel("N - tokens seen (log scale)")
    bottom.set_ylabel("local slope  d(log V)/d(log N)")
    bottom.set_title("The slope is the exponent - and it keeps falling",
                     fontsize=11, pad=8)
    bottom.set_ylim(0, 1.08)
    bottom.grid(True, which="major", linewidth=0.3, alpha=0.35)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_linear(grid, mean, path):
    figure, axes = plt.subplots(figsize=(10, 7))

    axes.plot(grid, mean, linewidth=2.0, color=COLOUR, label="observed V(N)")
    axes.fill_between(grid, 0, mean, color=COLOUR, alpha=0.15, linewidth=0)

    previous = None

    for marker in [1_000_000, 2_000_000, 3_000_000]:
        if marker > grid[-1]:
            continue

        value = float(np.interp(marker, grid, mean))
        axes.plot([marker], [value], marker="o", markersize=5, color="#111111")

        if previous is None:
            note = f"{marker / 1e6:.0f}M tokens\n{value:,.0f} types"
        else:
            note = (f"{marker / 1e6:.0f}M tokens\n{value:,.0f} types\n"
                    f"(+{value - previous:,.0f})")

        axes.annotate(note, xy=(marker, value), xytext=(10, -36),
                      textcoords="offset points", fontsize=8.5,
                      color="#333333")
        previous = value

    axes.set_xlabel("N - tokens seen (linear scale)")
    axes.set_ylabel("V - distinct word forms (linear scale)")
    axes.set_title("Vocabulary growth on linear axes\n"
                   "each additional million tokens adds fewer new words",
                   fontsize=12.5, pad=12)
    axes.legend(fontsize=9, loc="upper left")
    axes.grid(True, linewidth=0.3, alpha=0.3)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_rate(grid, mean, path):
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
    axes.set_title("Rate of vocabulary discovery\n"
                   f"a {first / last:.0f}x decline across the corpus",
                   fontsize=12.5, pad=12)
    axes.legend(fontsize=9)
    axes.grid(True, which="major", linewidth=0.3, alpha=0.35)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    grid, mean, spread = load_curve(CURVE_CSV)
    centres, slopes = local_slope(grid, mean)

    plot_main(grid, mean, spread, centres, slopes,
              os.path.join(OUT_DIR, "heaps_main.png"))
    plot_linear(grid, mean, os.path.join(OUT_DIR, "heaps_linear.png"))
    plot_rate(grid, mean, os.path.join(OUT_DIR, "heaps_discovery_rate.png"))

    print("local slope at selected N:")

    for target in [1e3, 1e4, 1e5, 1e6, 3e6]:
        if target > centres[-1]:
            continue

        print(f"  N = {int(target):>9,}   slope = "
              f"{float(np.interp(target, centres, slopes)):.3f}")

    print()
    print("wrote heaps_main.png, heaps_linear.png, heaps_discovery_rate.png")


if __name__ == "__main__":
    main()
