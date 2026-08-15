"""
Same two windows (first 1,000 and first 10,000 word forms) on LINEAR axes.

On linear axes a Zipf distribution is a hyperbola, not a straight line, so the
fitted power law is drawn as a curve rather than a line of best fit.
"""

import glob
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


CORPUS_DIR = "scripture_corpus"
OUT_DIR = "out/windows-linear"

WINDOWS = [1000, 10000]


def register_fonts():
    patterns = [
        "/usr/share/fonts/truetype/ezra/*.ttf",
        "/usr/share/fonts/opentype/fonts-hosny-amiri/*.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari*.ttf",
    ]

    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                font_manager.fontManager.addfont(path)
            except Exception:
                pass


register_fonts()

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAVE_BIDI = True
except ImportError:
    HAVE_BIDI = False


CORPORA = [
    ("tipitaka_all", "Pali Canon (Tipitaka)", "latin", "#8c6d3f"),
    ("hebrew_ot_whole", "Hebrew Bible (Tanakh)", "hebrew", "#8c4a3f"),
    ("greek_nt", "Greek New Testament", "greek", "#3f6b8c"),
    ("arabic_quran", "Qur'an", "arabic", "#3f8c5a"),
    ("sanskrit_gita", "Bhagavad Gita", "devanagari", "#8c7a3f"),
]

FONTS = {
    "hebrew": "Ezra SIL SR",
    "arabic": "Amiri",
    "devanagari": "Noto Sans Devanagari",
    "greek": "DejaVu Sans",
    "latin": "DejaVu Sans",
}


def shape(word, script):
    if not HAVE_BIDI:
        return word

    if script == "arabic":
        return get_display(arabic_reshaper.reshape(word))

    if script == "hebrew":
        return get_display(word)

    return word


def load_table(name):
    path = os.path.join(CORPUS_DIR, name + ".txt")

    with open(path, encoding="utf-8") as handle:
        tokens = handle.read().split()

    counts = Counter(tokens)
    table = []

    for rank, (word, count) in enumerate(counts.most_common(), start=1):
        table.append((rank, word, count))

    return table, len(tokens)


def plot_linear_window(table, total, limit, label, script, colour, path,
                       label_count):
    window = table[:limit]

    ranks = np.array([row[0] for row in window], dtype=float)
    frequencies = np.array([row[2] for row in window], dtype=float)

    slope, intercept = np.polyfit(np.log10(ranks), np.log10(frequencies), 1)
    fitted = 10 ** (intercept + slope * np.log10(ranks))

    figure, axes = plt.subplots(figsize=(12, 7))

    axes.plot(ranks, frequencies, linewidth=1.4, color=colour,
              label=f"observed frequency (ranks 1-{limit:,})")
    axes.fill_between(ranks, frequencies, color=colour, alpha=0.22)

    axes.plot(ranks, fitted, linestyle="--", linewidth=1.4, color="#111111",
              label=f"fitted power law (exponent {slope:.3f})")

    ideal = frequencies[0] / ranks
    axes.plot(ranks, ideal, linestyle=":", linewidth=1.4, color="#777777",
              label="ideal Zipf (exponent -1)")

    font = font_manager.FontProperties(family=FONTS[script], size=9)

    for index in range(label_count):
        rank = index + 1

        if rank > len(window):
            break

        word = shape(window[rank - 1][1], script)
        frequency = window[rank - 1][2]

        horizontal = 34 + (index % 4) * 62
        vertical = 6 + (index // 4) * 4

        axes.annotate(
            word,
            xy=(rank, frequency),
            xytext=(horizontal, vertical),
            textcoords="offset points",
            fontproperties=font,
            color="#222222",
            arrowprops={
                "arrowstyle": "-",
                "linewidth": 0.6,
                "color": "#888888",
                "shrinkA": 1,
                "shrinkB": 2,
            },
        )
        axes.plot([rank], [frequency], marker="o", markersize=3.6,
                  color="#111111")

    coverage = 100.0 * sum(row[2] for row in window) / total

    axes.set_xlabel("rank of word form (linear scale)")
    axes.set_ylabel("frequency (linear scale)")
    axes.set_title(
        f"{label} - first {limit:,} word forms, linear axes\n"
        f"these cover {coverage:.1f}% of all {total:,} tokens",
        fontsize=12, pad=14,
    )
    axes.set_xlim(0, limit)
    axes.set_ylim(0, frequencies[0] * 1.08)
    axes.legend(fontsize=9)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.grid(True, linewidth=0.3, alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

    return coverage


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for name, label, script, colour in CORPORA:
        corpus_path = os.path.join(CORPUS_DIR, name + ".txt")

        if not os.path.exists(corpus_path):
            continue

        table, total = load_table(name)

        for limit in WINDOWS:
            if len(table) < limit:
                continue

            if limit <= 1000:
                label_count = 12
            else:
                label_count = 6

            path = os.path.join(OUT_DIR, f"{name}_top{limit}_linear.png")
            coverage = plot_linear_window(table, total, limit, label, script,
                                          colour, path, label_count)

            print(f"{label:<26} top {limit:>6,}  covers {coverage:>5.1f}%")

    print()
    print("wrote plots to", OUT_DIR)


if __name__ == "__main__":
    main()
