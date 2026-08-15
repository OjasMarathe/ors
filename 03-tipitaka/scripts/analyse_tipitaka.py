"""
Zipf's law on the Pali Canon (Tipitaka).

Source: SuttaCentral bilara-data, root/pli/ms - the Mahasangiti edition of the
Tipitaka in romanised Pali, segmented into addressable units.

Builds four corpora (whole canon plus the three baskets separately), runs the
same analysis used for the Bible/Quran/Gita, and adds Pali to the cross-scripture
comparison plot.
"""

import csv
import glob
import json
import os
import re
import unicodedata
from collections import Counter

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = "scripture_src/suttacentral_bilara-data/root/pli/ms"
CORPUS_DIR = "scripture_corpus"
OUT_DIR = "out/tipitaka-zipf"

# Pali in romanised form: Latin letters plus the standard diacritics.
PALI_TOKEN = re.compile(r"[a-zāīūṁṃṅñṭḍṇḷṛśṣ]+")

# Peyyala markers: the canon abbreviates repeated passages rather than writing
# them out. These are editorial, not words.
PEYYALA = {"pe", "la"}

BASKETS = [
    ("tipitaka_vinaya", "vinaya", "Vinaya Pitaka\n(monastic rules)", "#7a5c3f"),
    ("tipitaka_sutta", "sutta", "Sutta Pitaka\n(discourses)", "#3f6b8c"),
    ("tipitaka_abhidhamma", "abhidhamma",
     "Abhidhamma Pitaka\n(systematic analysis)", "#8c3f5a"),
]


# Some segments carry inline HTML (<b>, <i>, <j>) around lemma citations.
# Left in place, the tag letters get tokenised as Pali words.
HTML_TAG = re.compile(r"<[^>]*>")


def tokenise(text):
    stripped = HTML_TAG.sub(" ", text)
    lowered = unicodedata.normalize("NFC", stripped).lower()
    tokens = PALI_TOKEN.findall(lowered)
    return [token for token in tokens if token not in PEYYALA]


def load_basket(subdirectory):
    """Read every segmented JSON file under one basket and return its tokens."""
    pattern = os.path.join(ROOT, subdirectory, "**", "*.json")
    paths = sorted(glob.glob(pattern, recursive=True))
    tokens = []

    for path in paths:
        with open(path, encoding="utf-8") as handle:
            try:
                segments = json.load(handle)
            except json.JSONDecodeError:
                continue

        for value in segments.values():
            if isinstance(value, str):
                tokens.extend(tokenise(value))

    return tokens, len(paths)


def write_corpus(name, tokens):
    os.makedirs(CORPUS_DIR, exist_ok=True)
    path = os.path.join(CORPUS_DIR, name + ".txt")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(" ".join(tokens))

    return path


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


def mle_fit(table):
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


def plot_top_words(table, total, path, top_n=20):
    top = table[:top_n][::-1]
    words = [row[1] for row in top]
    counts = [row[2] for row in top]

    figure, axes = plt.subplots(figsize=(9, 0.34 * top_n + 1.8))
    positions = np.arange(len(words))
    axes.barh(positions, counts, color="#8c6d3f", edgecolor="none")

    axes.set_yticks(positions)
    axes.set_yticklabels(words, fontsize=10)
    axes.set_xlabel("frequency")
    axes.set_title(f"{top_n} most frequent word forms - Pali Canon (Tipitaka)",
                   fontsize=12, pad=12)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)

    limit = max(counts)
    axes.set_xlim(0, limit * 1.2)

    for position, count in zip(positions, counts):
        axes.text(count + limit * 0.012, position,
                  f"{count:,} ({100.0 * count / total:.2f}%)",
                  va="center", fontsize=7.5, color="#555555")

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_basket_curves(results, path):
    figure, axes = plt.subplots(figsize=(9, 7))

    order = [
        ("tipitaka_all", "whole canon", "#222222"),
        ("tipitaka_vinaya", "Vinaya (monastic rules)", "#7a5c3f"),
        ("tipitaka_sutta", "Sutta (discourses)", "#3f6b8c"),
        ("tipitaka_abhidhamma", "Abhidhamma (systematic analysis)", "#8c3f5a"),
    ]

    for name, label, colour in order:
        entry = results[name]
        table = entry["table"]
        total = entry["total"]
        slope = entry["slope"]

        ranks = np.array([row[0] for row in table], dtype=float)
        relative = np.array([row[2] for row in table], dtype=float) / total

        axes.loglog(ranks, relative, linewidth=1.4, color=colour, alpha=0.9,
                    label=f"{label}  (slope {slope:.2f})")

    reference = np.logspace(0, 5.2, 60)
    axes.loglog(reference, 0.05 / reference, linestyle=":", color="#333333",
                linewidth=1.4, label="ideal Zipf (slope -1)")

    axes.set_xlabel("rank of word form (log scale)")
    axes.set_ylabel("relative frequency (log scale)")
    axes.set_title("Zipf's law within the Pali Canon, by basket",
                   fontsize=12.5, pad=12)
    axes.legend(fontsize=8.5, loc="lower left")
    axes.grid(True, which="major", linewidth=0.3, alpha=0.4)

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_all_scriptures(pali_entry, path):
    """Redraw the cross-scripture comparison with Pali added."""
    others = [
        ("hebrew_ot_whole", "Hebrew Bible", "#8c4a3f"),
        ("greek_nt", "Greek New Testament", "#3f6b8c"),
        ("arabic_quran", "Qur'an", "#3f8c5a"),
        ("sanskrit_gita", "Bhagavad Gita", "#8c7a3f"),
    ]

    figure, axes = plt.subplots(figsize=(9.5, 7))

    for name, label, colour in others:
        corpus_path = os.path.join(CORPUS_DIR, name + ".txt")

        if not os.path.exists(corpus_path):
            continue

        with open(corpus_path, encoding="utf-8") as handle:
            tokens = handle.read().split()

        table = rank_table(tokens)
        slope, _ = ols_slope(table)

        ranks = np.array([row[0] for row in table], dtype=float)
        relative = np.array([row[2] for row in table], dtype=float) / len(tokens)

        axes.loglog(ranks, relative, linewidth=1.3, color=colour, alpha=0.75,
                    label=f"{label}  ({slope:.2f})")

    table = pali_entry["table"]
    total = pali_entry["total"]
    slope = pali_entry["slope"]

    ranks = np.array([row[0] for row in table], dtype=float)
    relative = np.array([row[2] for row in table], dtype=float) / total
    axes.loglog(ranks, relative, linewidth=2.0, color="#111111",
                label=f"Pali Canon  ({slope:.2f})")

    reference = np.logspace(0, 5.2, 60)
    axes.loglog(reference, 0.06 / reference, linestyle=":", color="#666666",
                linewidth=1.4, label="ideal Zipf (-1)")

    axes.set_xlabel("rank of word form (log scale)")
    axes.set_ylabel("relative frequency (log scale)")
    axes.set_title("Zipf's law across five scriptures in their original languages",
                   fontsize=12.5, pad=12)
    axes.legend(fontsize=8.5, loc="lower left")
    axes.grid(True, which="major", linewidth=0.3, alpha=0.4)

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {}
    summary_rows = []
    all_tokens = []

    for name, subdirectory, label, colour in BASKETS:
        tokens, file_count = load_basket(subdirectory)
        all_tokens.extend(tokens)
        write_corpus(name, tokens)
        results[name] = {"tokens": tokens, "files": file_count}
        print(f"{subdirectory:<12} files={file_count:>5}  tokens={len(tokens):>9,}")

    write_corpus("tipitaka_all", all_tokens)
    results["tipitaka_all"] = {"tokens": all_tokens, "files": 0}
    print(f"{'ALL':<12} files={'':>5}  tokens={len(all_tokens):>9,}")
    print()

    for name in ["tipitaka_all", "tipitaka_vinaya", "tipitaka_sutta",
                 "tipitaka_abhidhamma"]:
        tokens = results[name]["tokens"]
        table = rank_table(tokens)
        slope, _ = ols_slope(table)
        mle = mle_fit(table)

        total = len(tokens)
        types = len(table)
        hapax = sum(1 for _, _, count in table if count == 1)

        results[name].update({"table": table, "total": total, "slope": slope})

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

        write_rank_table(table, total,
                         os.path.join(OUT_DIR, f"rank_frequency_{name}.csv"))

    with open(os.path.join(OUT_DIR, "tipitaka_summary.csv"),
              "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    header = f"{'corpus':<24}{'tokens':>11}{'types':>9}{'TTR':>8}" \
             f"{'hapax%':>9}{'OLS':>8}{'alpha':>8}{'KS':>8}"
    print(header)
    print("-" * len(header))

    for row in summary_rows:
        print(f"{row['corpus']:<24}{row['tokens']:>11,}{row['types']:>9,}"
              f"{row['type_token_ratio']:>8}{row['hapax_pct']:>9}"
              f"{row['ols_slope']:>8}{row['mle_alpha']:>8}{row['ks_distance']:>8}")

    whole = results["tipitaka_all"]
    plot_top_words(whole["table"], whole["total"],
                   os.path.join(OUT_DIR, "top_words_tipitaka.png"))
    plot_basket_curves(results, os.path.join(OUT_DIR, "zipf_tipitaka_baskets.png"))
    plot_all_scriptures(whole, os.path.join(OUT_DIR, "zipf_all_scriptures.png"))

    print()
    print("wrote plots and tables to", OUT_DIR)


if __name__ == "__main__":
    main()
