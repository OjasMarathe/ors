"""
Plot the FULL vocabulary of each corpus - every distinct word form, not a top-N slice.

For each corpus this produces:
    <name>_all_words.png         log-log rank vs frequency, every type plotted,
                                 with a log-spaced sample of words labelled
    <name>_all_words_linear.png  the same data on linear axes

Labelling all 153,693 Pali types is physically impossible on a page, so the
labels are a sample drawn at log-spaced ranks. Every type is still plotted as a
point - nothing is truncated.
"""

import os
import unicodedata
from collections import Counter

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


CORPUS_DIR = "scripture_corpus"
OUT_DIR = "out/all-words"


def register_fonts():
    import glob

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


def pick_label_ranks(vocabulary_size, count):
    """Choose ranks spread evenly in log space, so labels don't pile up."""
    positions = np.logspace(0, np.log10(vocabulary_size), count)
    chosen = sorted({int(round(position)) for position in positions})
    return [rank for rank in chosen if 1 <= rank <= vocabulary_size]


def plot_log(table, total, label, script, colour, path, label_count=22):
    ranks = np.array([row[0] for row in table], dtype=float)
    frequencies = np.array([row[2] for row in table], dtype=float)

    slope, intercept = np.polyfit(np.log10(ranks), np.log10(frequencies), 1)

    figure, axes = plt.subplots(figsize=(12, 8.5))

    axes.loglog(ranks, frequencies, marker=".", linestyle="none",
                markersize=1.6, color=colour, alpha=0.45, rasterized=True,
                label=f"all {len(table):,} distinct word forms")

    fitted = 10 ** (intercept + slope * np.log10(ranks))
    axes.loglog(ranks, fitted, linestyle="--", linewidth=1.3, color="#111111",
                label=f"least-squares fit (slope {slope:.3f})")

    ideal = frequencies[0] / ranks
    axes.loglog(ranks, ideal, linestyle=":", linewidth=1.3, color="#777777",
                label="ideal Zipf (slope -1)")

    font = font_manager.FontProperties(family=FONTS[script], size=8.5)

    label_ranks = pick_label_ranks(len(table), label_count)

    for index, rank in enumerate(label_ranks):
        word = shape(table[rank - 1][1], script)
        frequency = table[rank - 1][2]

        if index % 2 == 0:
            offset = (7, 8)
        else:
            offset = (7, -14)

        axes.annotate(
            word,
            xy=(rank, frequency),
            xytext=offset,
            textcoords="offset points",
            fontproperties=font,
            color="#222222",
            alpha=0.9,
        )
        axes.plot([rank], [frequency], marker="o", markersize=3.2,
                  color="#111111", alpha=0.75)

    axes.set_ylim(frequencies.min() * 0.45, frequencies.max() * 4)
    axes.set_xlim(0.7, len(table) * 3.5)

    hapax = sum(1 for _, _, count in table if count == 1)

    axes.set_xlabel("rank of word form (log scale)")
    axes.set_ylabel("frequency (log scale)")
    axes.set_title(
        f"{label} - complete vocabulary\n"
        f"{total:,} tokens, {len(table):,} distinct forms, "
        f"{hapax:,} occur exactly once ({100.0 * hapax / len(table):.1f}%)",
        fontsize=12.5, pad=14,
    )
    axes.legend(fontsize=9, loc="lower left")
    axes.grid(True, which="major", linewidth=0.3, alpha=0.35)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

    return slope


def plot_linear(table, total, label, colour, path):
    ranks = np.array([row[0] for row in table], dtype=float)
    frequencies = np.array([row[2] for row in table], dtype=float)

    figure, axes = plt.subplots(figsize=(11, 6))
    axes.plot(ranks, frequencies, linewidth=0.9, color=colour)
    axes.fill_between(ranks, frequencies, color=colour, alpha=0.25)

    axes.set_xlabel("rank of word form (linear scale)")
    axes.set_ylabel("frequency (linear scale)")
    axes.set_title(
        f"{label} - the same data on linear axes\n"
        f"every one of the {len(table):,} forms is plotted; "
        "the curve is indistinguishable from the axes almost immediately",
        fontsize=11.5, pad=12,
    )
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for name, label, script, colour in CORPORA:
        path = os.path.join(CORPUS_DIR, name + ".txt")

        if not os.path.exists(path):
            print(f"skipping {name} - corpus file missing")
            continue

        table, total = load_table(name)

        log_path = os.path.join(OUT_DIR, f"{name}_all_words.png")
        linear_path = os.path.join(OUT_DIR, f"{name}_all_words_linear.png")

        slope = plot_log(table, total, label, script, colour, log_path)
        plot_linear(table, total, label, colour, linear_path)

        print(f"{label:<26} types={len(table):>8,}  slope={slope:.3f}")

    print()
    print("wrote plots to", OUT_DIR)


if __name__ == "__main__":
    main()
