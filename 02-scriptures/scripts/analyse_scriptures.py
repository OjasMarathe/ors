"""
Zipf's law across the Bible, Quran and Bhagavad Gita in their original languages.

Produces:
    top_words_scriptures.png     top 15 words per scripture, native script
    zipf_scriptures_loglog.png   rank-frequency curves, all corpora
    morphology_effect.png        Hebrew whole-word vs morpheme-split
    scripture_summary.csv        token/type/hapax/exponent table
    rank_frequency_<corpus>.csv  full rank tables
"""

import csv
import glob
import os
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


CORPUS_DIR = "scripture_corpus"
OUT_DIR = "out/scripture-zipf"

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAVE_BIDI = True
except ImportError:
    HAVE_BIDI = False


def register_fonts():
    """matplotlib caches its font list, so newly installed fonts need adding."""
    patterns = [
        "/usr/share/fonts/truetype/ezra/*.ttf",
        "/usr/share/fonts/opentype/fonts-hosny-amiri/*.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari*.ttf",
        "/usr/share/fonts/truetype/lohit-devanagari/*.ttf",
    ]

    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                font_manager.fontManager.addfont(path)
            except Exception:
                pass


register_fonts()


FONTS = {
    "hebrew": "Ezra SIL SR",
    "arabic": "Amiri",
    "devanagari": "Noto Sans Devanagari",
    "greek": "DejaVu Sans",
    "latin": "DejaVu Sans",
}


CORPORA = [
    ("hebrew_ot_whole", "Hebrew Bible (Tanakh)\nWestminster Leningrad Codex",
     "hebrew", "#8c4a3f"),
    ("greek_nt", "Greek New Testament\nSBLGNT", "greek", "#3f6b8c"),
    ("arabic_quran", "Qur'an\nArabic, diacritics stripped", "arabic", "#3f8c5a"),
    ("sanskrit_gita", "Bhagavad Gita\nSanskrit, Devanagari", "devanagari", "#8c7a3f"),
]


def shape(word, script):
    """Apply Arabic joining and right-to-left reordering where needed."""
    if not HAVE_BIDI:
        return word

    if script == "arabic":
        reshaped = arabic_reshaper.reshape(word)
        return get_display(reshaped)

    if script == "hebrew":
        return get_display(word)

    return word


def load_tokens(name):
    path = os.path.join(CORPUS_DIR, name + ".txt")

    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    return text.split()


def rank_table(tokens):
    counts = Counter(tokens)
    table = []

    for rank, (word, count) in enumerate(counts.most_common(), start=1):
        table.append((rank, word, count))

    return table


def ols_slope(table):
    ranks = np.array([row[0] for row in table], dtype=float)
    frequencies = np.array([row[2] for row in table], dtype=float)
    slope, intercept = np.polyfit(np.log10(ranks), np.log10(frequencies), 1)
    return slope, intercept


def mle_alpha(table):
    try:
        import powerlaw
    except ImportError:
        return None

    frequencies = np.array([row[2] for row in table], dtype=float)
    fit = powerlaw.Fit(frequencies, discrete=True, verbose=False)
    return fit.power_law.alpha, fit.power_law.xmin, fit.power_law.D


def write_rank_table(table, total, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "word", "frequency", "relative_frequency"])

        for rank, word, count in table:
            writer.writerow([rank, word, count, count / total])


def plot_top_words_grid(results, path, top_n=15):
    figure, axes_grid = plt.subplots(2, 2, figsize=(13, 11))
    axes_list = axes_grid.flatten()

    for axes, (name, label, script, colour) in zip(axes_list, CORPORA):
        table = results[name]["table"]
        total = results[name]["total"]

        top = table[:top_n][::-1]
        words = [shape(row[1], script) for row in top]
        counts = [row[2] for row in top]

        positions = np.arange(len(words))
        axes.barh(positions, counts, color=colour, edgecolor="none")

        font = font_manager.FontProperties(family=FONTS[script], size=13)
        axes.set_yticks(positions)
        axes.set_yticklabels(words, fontproperties=font)

        limit = max(counts)
        axes.set_xlim(0, limit * 1.25)

        for position, count in zip(positions, counts):
            share = 100.0 * count / total
            axes.text(count + limit * 0.02, position,
                      f"{count:,} ({share:.1f}%)",
                      va="center", fontsize=7.5, color="#555555")

        axes.set_title(label, fontsize=10.5, pad=8)
        axes.set_xlabel("frequency", fontsize=8.5)
        axes.tick_params(axis="x", labelsize=7.5)
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

    figure.suptitle(
        f"{top_n} most frequent word forms - original-language scriptures",
        fontsize=13.5,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.965])
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_zipf(results, extra, path):
    figure, axes = plt.subplots(figsize=(9, 7))

    for name, label, script, colour in CORPORA:
        table = results[name]["table"]
        total = results[name]["total"]
        slope = results[name]["slope"]

        ranks = np.array([row[0] for row in table], dtype=float)
        frequencies = np.array([row[2] for row in table], dtype=float)
        relative = frequencies / total

        short = label.split("\n")[0]
        axes.loglog(ranks, relative, linewidth=1.4, color=colour, alpha=0.85,
                    label=f"{short}  (slope {slope:.2f})")

    if extra is not None:
        ranks, relative, slope = extra
        axes.loglog(ranks, relative, linewidth=1.2, color="#999999",
                    linestyle="-", alpha=0.8,
                    label=f"English (SC judgments)  (slope {slope:.2f})")

    reference_ranks = np.logspace(0, 4.8, 50)
    reference_values = 0.07 / reference_ranks
    axes.loglog(reference_ranks, reference_values, linestyle=":",
                color="#333333", linewidth=1.5, label="ideal Zipf (slope -1)")

    axes.set_xlabel("rank of word form (log scale)")
    axes.set_ylabel("relative frequency (log scale)")
    axes.set_title("Zipf's law across scriptures in their original languages",
                   fontsize=12.5, pad=12)
    axes.legend(fontsize=8.5, loc="lower left")
    axes.grid(True, which="major", linewidth=0.3, alpha=0.4)

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_morphology_effect(path):
    figure, axes = plt.subplots(figsize=(8.5, 6))

    settings = [
        ("hebrew_ot_whole", "orthographic words (prefixes attached)", "#8c4a3f"),
        ("hebrew_ot_morphemes", "morphemes split at prefix boundaries", "#3f6b8c"),
    ]

    for name, label, colour in settings:
        tokens = load_tokens(name)
        table = rank_table(tokens)
        slope, _ = ols_slope(table)

        ranks = np.array([row[0] for row in table], dtype=float)
        relative = np.array([row[2] for row in table], dtype=float) / len(tokens)

        axes.loglog(ranks, relative, linewidth=1.5, color=colour,
                    label=f"{label}\n  {len(set(tokens)):,} types, slope {slope:.2f}")

    reference_ranks = np.logspace(0, 4.7, 50)
    axes.loglog(reference_ranks, 0.07 / reference_ranks, linestyle=":",
                color="#333333", linewidth=1.4, label="ideal Zipf (slope -1)")

    axes.set_xlabel("rank (log scale)")
    axes.set_ylabel("relative frequency (log scale)")
    axes.set_title("How much of the Hebrew tail is morphology?", fontsize=12, pad=12)
    axes.legend(fontsize=8.5, loc="lower left")
    axes.grid(True, which="major", linewidth=0.3, alpha=0.4)

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def english_reference():
    """Rank curve for the SC judgment prose corpus, for comparison."""
    paths = sorted(glob.glob("judgment_corpus/*.txt"))

    if not paths:
        return None

    pattern = re.compile(r"[a-z]+")
    tokens = []

    for path in paths:
        with open(path, encoding="utf-8", errors="ignore") as handle:
            tokens.extend(pattern.findall(handle.read().lower()))

    table = rank_table(tokens)
    slope, _ = ols_slope(table)

    ranks = np.array([row[0] for row in table], dtype=float)
    relative = np.array([row[2] for row in table], dtype=float) / len(tokens)

    return ranks, relative, slope


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {}
    summary_rows = []

    all_names = [row[0] for row in CORPORA] + ["hebrew_ot_morphemes"]

    for name in all_names:
        tokens = load_tokens(name)
        table = rank_table(tokens)
        slope, _ = ols_slope(table)

        total = len(tokens)
        types = len(table)
        hapax = sum(1 for _, _, count in table if count == 1)

        mle = mle_alpha(table)

        results[name] = {"table": table, "total": total, "slope": slope}

        summary_rows.append({
            "corpus": name,
            "tokens": total,
            "types": types,
            "type_token_ratio": round(types / total, 4),
            "hapax": hapax,
            "hapax_pct": round(100.0 * hapax / types, 1),
            "ols_slope": round(slope, 3),
            "mle_alpha": round(mle[0], 3) if mle else "",
            "mle_xmin": int(mle[1]) if mle else "",
            "ks_distance": round(mle[2], 4) if mle else "",
        })

        write_rank_table(
            table, total,
            os.path.join(OUT_DIR, f"rank_frequency_{name}.csv"),
        )

    with open(os.path.join(OUT_DIR, "scripture_summary.csv"),
              "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    header = f"{'corpus':<24}{'tokens':>10}{'types':>9}{'TTR':>8}" \
             f"{'hapax%':>9}{'OLS':>8}{'alpha':>8}{'KS':>8}"
    print(header)
    print("-" * len(header))

    for row in summary_rows:
        print(f"{row['corpus']:<24}{row['tokens']:>10,}{row['types']:>9,}"
              f"{row['type_token_ratio']:>8}{row['hapax_pct']:>9}"
              f"{row['ols_slope']:>8}{row['mle_alpha']:>8}{row['ks_distance']:>8}")

    plot_top_words_grid(results, os.path.join(OUT_DIR, "top_words_scriptures.png"))
    plot_zipf(results, english_reference(),
              os.path.join(OUT_DIR, "zipf_scriptures_loglog.png"))
    plot_morphology_effect(os.path.join(OUT_DIR, "morphology_effect.png"))

    print()
    print("bidi/reshaping available:", HAVE_BIDI)
    print("wrote plots and tables to", OUT_DIR)


if __name__ == "__main__":
    main()
