"""
Plot most-frequent words and Zipf curves for two Supreme Court of India corpora.

Corpus A : judgment prose  (122 documents of actual judgment text)
Corpus B : case titles     (43,495 case titles + judge names from the metadata)
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


TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")

OUT_DIR = "out/sc-zipf/results"


def tokenise(text):
    lowered = text.lower()
    tokens = TOKEN_PATTERN.findall(lowered)
    return tokens


def load_prose_tokens(corpus_dir):
    paths = sorted(glob.glob(os.path.join(corpus_dir, "*.txt")))
    tokens = []

    for path in paths:
        with open(path, encoding="utf-8", errors="ignore") as handle:
            tokens.extend(tokenise(handle.read()))

    return tokens, len(paths)


def load_title_tokens(metadata_path):
    tokens = []
    rows = 0

    with open(metadata_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            rows += 1
            for field in ("title", "judge"):
                tokens.extend(tokenise(row.get(field) or ""))

    return tokens, rows


def rank_table(tokens):
    counts = Counter(tokens)
    ordered = counts.most_common()
    table = []

    for rank, (word, count) in enumerate(ordered, start=1):
        table.append((rank, word, count))

    return table


def write_table(table, total_tokens, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "word", "frequency", "relative_frequency"])

        for rank, word, count in table:
            writer.writerow([rank, word, count, count / total_tokens])


def plot_top_words(table, total_tokens, title, path, top_n, colour):
    top = table[:top_n]
    words = [row[1] for row in top][::-1]
    counts = [row[2] for row in top][::-1]
    shares = [100.0 * c / total_tokens for c in counts]

    figure, axes = plt.subplots(figsize=(9, 0.32 * top_n + 1.8))
    positions = np.arange(len(words))
    axes.barh(positions, counts, color=colour, edgecolor="none")

    axes.set_yticks(positions)
    axes.set_yticklabels(words, fontsize=9)
    axes.set_xlabel("frequency (count of occurrences)")
    axes.set_title(title, fontsize=12, pad=12)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)

    limit = max(counts)
    axes.set_xlim(0, limit * 1.18)

    for position, count, share in zip(positions, counts, shares):
        axes.text(count + limit * 0.012, position,
                  f"{count:,}  ({share:.2f}%)",
                  va="center", fontsize=7.5, color="#444444")

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_zipf(datasets, path):
    figure, axes = plt.subplots(figsize=(8, 6))

    for label, table, colour in datasets:
        ranks = np.array([row[0] for row in table], dtype=float)
        frequencies = np.array([row[2] for row in table], dtype=float)

        slope, intercept = np.polyfit(np.log10(ranks), np.log10(frequencies), 1)

        axes.loglog(ranks, frequencies, marker=".", linestyle="none",
                    markersize=2.2, color=colour, alpha=0.55,
                    label=f"{label}  (OLS slope {slope:.2f})")

        fitted = 10 ** (intercept + slope * np.log10(ranks))
        axes.loglog(ranks, fitted, linestyle="--", linewidth=1.1, color=colour)

    reference_ranks = np.array([1, 10 ** 4.6])
    reference_values = 20000 / reference_ranks
    axes.loglog(reference_ranks, reference_values, linestyle=":",
                color="#666666", linewidth=1.4, label="ideal Zipf (slope = -1)")

    axes.set_xlabel("rank of word (log scale)")
    axes.set_ylabel("frequency (log scale)")
    axes.set_title("Zipf's law - Supreme Court of India", fontsize=12, pad=12)
    axes.legend(fontsize=8.5, loc="lower left")
    axes.grid(True, which="major", linewidth=0.3, alpha=0.4)

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def summarise(name, table, total_tokens, documents):
    vocabulary = len(table)
    hapax = sum(1 for _, _, count in table if count == 1)

    print()
    print(f"--- {name} ---")
    print(f"documents       : {documents:,}")
    print(f"tokens          : {total_tokens:,}")
    print(f"vocabulary      : {vocabulary:,}")
    print(f"hapax legomena  : {hapax:,} ({100.0 * hapax / vocabulary:.1f}%)")

    ranks = np.array([row[0] for row in table], dtype=float)
    frequencies = np.array([row[2] for row in table], dtype=float)
    slope, _ = np.polyfit(np.log10(ranks), np.log10(frequencies), 1)
    print(f"OLS log-log slope: {slope:.3f}")

    print("top 12:")
    for rank, word, count in table[:12]:
        print(f"  {rank:>3}  {word:<14} {count:>9,}  {100.0*count/total_tokens:>6.2f}%")

    return slope


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    prose_tokens, prose_docs = load_prose_tokens("judgment_corpus")
    prose_table = rank_table(prose_tokens)

    title_tokens, title_rows = load_title_tokens(
        "out/sc-zipf/sc_judgments_metadata.csv"
    )
    title_table = rank_table(title_tokens)

    summarise("Judgment prose", prose_table, len(prose_tokens), prose_docs)
    summarise("Case titles", title_table, len(title_tokens), title_rows)

    write_table(prose_table, len(prose_tokens),
                os.path.join(OUT_DIR, "rank_frequency_judgment_prose.csv"))
    write_table(title_table, len(title_tokens),
                os.path.join(OUT_DIR, "rank_frequency_case_titles.csv"))

    plot_top_words(
        prose_table,
        len(prose_tokens),
        "30 most frequent words - SC of India judgment text",
        os.path.join(OUT_DIR, "top_words_judgment_prose.png"),
        top_n=30,
        colour="#2f6f8f",
    )

    plot_top_words(
        title_table,
        len(title_tokens),
        "30 most frequent words - SC of India case titles (43,495 cases)",
        os.path.join(OUT_DIR, "top_words_case_titles.png"),
        top_n=30,
        colour="#8f5a2f",
    )

    plot_zipf(
        [
            ("judgment prose", prose_table, "#2f6f8f"),
            ("case titles", title_table, "#8f5a2f"),
        ],
        os.path.join(OUT_DIR, "zipf_loglog_both.png"),
    )

    print()
    print("wrote plots and rank-frequency tables to", OUT_DIR)


if __name__ == "__main__":
    main()
