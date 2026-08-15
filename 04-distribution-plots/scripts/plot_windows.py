"""
Zoomed rank-frequency plots: the first 1,000 and the first 10,000 word forms.

Each window gets its own least-squares fit, computed only on the points inside
that window. Comparing those slopes against the whole-corpus slope shows how
strongly the fitted Zipf exponent depends on where you cut the tail.
"""

import csv
import glob
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


CORPUS_DIR = "scripture_corpus"
OUT_DIR = "out/windows"

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


def fit(ranks, frequencies):
    slope, intercept = np.polyfit(np.log10(ranks), np.log10(frequencies), 1)
    return slope, intercept


def pick_label_ranks(limit, count):
    positions = np.logspace(0, np.log10(limit), count)
    chosen = sorted({int(round(position)) for position in positions})
    return [rank for rank in chosen if 1 <= rank <= limit]


def plot_window(table, total, limit, label, script, colour, path,
                whole_slope, label_count=18):
    window = table[:limit]

    ranks = np.array([row[0] for row in window], dtype=float)
    frequencies = np.array([row[2] for row in window], dtype=float)

    slope, intercept = fit(ranks, frequencies)

    figure, axes = plt.subplots(figsize=(11.5, 7.5))

    axes.loglog(ranks, frequencies, marker=".", linestyle="none",
                markersize=3.0, color=colour, alpha=0.6,
                label=f"ranks 1-{limit:,} ({len(window):,} word forms)")

    fitted = 10 ** (intercept + slope * np.log10(ranks))
    axes.loglog(ranks, fitted, linestyle="--", linewidth=1.5, color="#111111",
                label=f"fit on this window: slope {slope:.3f}")

    ideal = frequencies[0] / ranks
    axes.loglog(ranks, ideal, linestyle=":", linewidth=1.4, color="#777777",
                label="ideal Zipf (slope -1)")

    font = font_manager.FontProperties(family=FONTS[script], size=8.5)
    label_ranks = pick_label_ranks(limit, label_count)

    for index, rank in enumerate(label_ranks):
        word = shape(table[rank - 1][1], script)
        frequency = table[rank - 1][2]

        if index % 2 == 0:
            offset = (7, 8)
        else:
            offset = (7, -14)

        axes.annotate(word, xy=(rank, frequency), xytext=offset,
                      textcoords="offset points", fontproperties=font,
                      color="#222222", alpha=0.9)
        axes.plot([rank], [frequency], marker="o", markersize=3.4,
                  color="#111111", alpha=0.8)

    coverage = 100.0 * sum(row[2] for row in window) / total

    axes.set_xlabel("rank of word form (log scale)")
    axes.set_ylabel("frequency (log scale)")
    axes.set_title(
        f"{label} - first {limit:,} word forms\n"
        f"these account for {coverage:.1f}% of all {total:,} tokens   |   "
        f"whole-corpus slope was {whole_slope:.3f}",
        fontsize=12, pad=14,
    )
    axes.legend(fontsize=9, loc="lower left")
    axes.grid(True, which="major", linewidth=0.3, alpha=0.35)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

    return slope, coverage


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary_rows = []

    for name, label, script, colour in CORPORA:
        corpus_path = os.path.join(CORPUS_DIR, name + ".txt")

        if not os.path.exists(corpus_path):
            continue

        table, total = load_table(name)

        all_ranks = np.array([row[0] for row in table], dtype=float)
        all_frequencies = np.array([row[2] for row in table], dtype=float)
        whole_slope, _ = fit(all_ranks, all_frequencies)

        row = {
            "corpus": name,
            "types_total": len(table),
            "slope_all": round(whole_slope, 3),
        }

        for limit in WINDOWS:
            if len(table) < limit:
                row[f"slope_top{limit}"] = ""
                row[f"coverage_top{limit}"] = ""
                continue

            path = os.path.join(OUT_DIR, f"{name}_top{limit}.png")
            slope, coverage = plot_window(table, total, limit, label, script,
                                          colour, path, whole_slope)

            row[f"slope_top{limit}"] = round(slope, 3)
            row[f"coverage_top{limit}"] = round(coverage, 1)

        summary_rows.append(row)

    fieldnames = ["corpus", "types_total", "slope_top1000", "coverage_top1000",
                  "slope_top10000", "coverage_top10000", "slope_all"]

    with open(os.path.join(OUT_DIR, "window_slopes.csv"),
              "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in summary_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    header = f"{'corpus':<20}{'top1k':>9}{'cov%':>8}{'top10k':>9}{'cov%':>8}{'all':>9}"
    print(header)
    print("-" * len(header))

    for row in summary_rows:
        print(f"{row['corpus']:<20}"
              f"{str(row.get('slope_top1000', '')):>9}"
              f"{str(row.get('coverage_top1000', '')):>8}"
              f"{str(row.get('slope_top10000', '')):>9}"
              f"{str(row.get('coverage_top10000', '')):>8}"
              f"{row['slope_all']:>9}")

    print()
    print("wrote plots to", OUT_DIR)


if __name__ == "__main__":
    main()
