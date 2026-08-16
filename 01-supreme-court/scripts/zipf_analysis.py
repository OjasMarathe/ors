"""
Zipf's law analysis over the Supreme Court of India judgment corpus.

Two modes:
    --source corpus     tokenise the extracted judgment texts in ./corpus/
    --source metadata   tokenise case titles + party names from the metadata CSV
                        (works immediately, no downloads needed)

Usage:
    pip install matplotlib numpy powerlaw
    python zipf_analysis.py --source corpus

Outputs:
    rank_frequency.csv    rank, word, frequency, relative frequency
    zipf_loglog.png       rank-frequency plot on log-log axes
    zipf_ccdf.png         complementary CDF of word frequencies
    heaps.png             vocabulary growth vs tokens seen
"""

import argparse
import csv
import glob
import math
import os
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")

# Boilerplate that appears in the header/footer of nearly every judgment PDF.
# Keep this list visible in the report - removing it changes the head of the
# distribution noticeably, which is itself a result worth showing.
BOILERPLATE = {
    "reportable",
    "nonreportable",
    "digitally",
    "signed",
    "verified",
    "www",
    "sci",
    "gov",
    "in",
}


def tokenise(text):
    """Lowercase, then pull out alphabetic word tokens."""
    lowered = text.lower()
    tokens = TOKEN_PATTERN.findall(lowered)
    return tokens


def load_corpus_tokens(corpus_dir):
    """Stream tokens out of every .txt file in the corpus directory."""
    paths = sorted(glob.glob(os.path.join(corpus_dir, "*.txt")))

    if not paths:
        raise SystemExit(
            f"no .txt files in {corpus_dir}/ - run download_judgments.py first"
        )

    print(f"reading {len(paths)} judgment texts ...")
    all_tokens = []

    for index, path in enumerate(paths, start=1):
        with open(path, encoding="utf-8", errors="ignore") as handle:
            text = handle.read()

        all_tokens.extend(tokenise(text))

        if index % 200 == 0:
            print(f"  {index}/{len(paths)} files, {len(all_tokens)} tokens so far")

    return all_tokens


def load_metadata_tokens(metadata_path):
    """Tokenise case titles and judge names from the metadata CSV."""
    print(f"reading {metadata_path} ...")
    all_tokens = []

    with open(metadata_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            for field in ("title", "judge"):
                value = row.get(field) or ""
                all_tokens.extend(tokenise(value))

    return all_tokens


def build_rank_table(tokens, drop_boilerplate):
    """Count tokens and return a rank-ordered list of (rank, word, count)."""
    counts = Counter(tokens)

    if drop_boilerplate:
        for word in BOILERPLATE:
            counts.pop(word, None)

    ordered = counts.most_common()
    table = []

    for rank, (word, count) in enumerate(ordered, start=1):
        table.append((rank, word, count))

    return table


def fit_ols(ranks, frequencies):
    """
    Naive log-log least squares fit.

    This is the standard approach in textbooks and it is biased by the noisy
    tail - reported here only so the report can show the discrepancy against
    the MLE estimate below.
    """
    log_ranks = np.log10(ranks)
    log_frequencies = np.log10(frequencies)
    slope, intercept = np.polyfit(log_ranks, log_frequencies, 1)
    return slope, intercept


def fit_mle(frequencies):
    """
    Clauset-Shalizi-Newman maximum likelihood fit with a fitted x_min.

    Returns (alpha, x_min, ks_distance) or None if the powerlaw package
    is not installed.
    """
    try:
        import powerlaw
    except ImportError:
        print("  (install `powerlaw` for the MLE fit: pip install powerlaw)")
        return None

    result = powerlaw.Fit(frequencies, discrete=True, verbose=False)
    alpha = result.power_law.alpha
    x_min = result.power_law.xmin
    ks_distance = result.power_law.D

    comparison = result.distribution_compare("power_law", "lognormal")
    log_likelihood_ratio = comparison[0]
    p_value = comparison[1]

    return alpha, x_min, ks_distance, log_likelihood_ratio, p_value


def heaps_curve(tokens, sample_points):
    """Vocabulary size as a function of tokens seen, at evenly spaced points."""
    step = max(1, len(tokens) // sample_points)
    seen = set()
    xs = []
    ys = []

    for index, token in enumerate(tokens, start=1):
        seen.add(token)

        if index % step == 0:
            xs.append(index)
            ys.append(len(seen))

    return np.array(xs), np.array(ys)


def plot_loglog(ranks, frequencies, slope, intercept, path):
    """Rank-frequency scatter on log-log axes with the OLS line drawn on top."""
    figure, axes = plt.subplots(figsize=(7, 5))
    axes.loglog(ranks, frequencies, marker=".", linestyle="none", markersize=2)

    fitted = 10 ** (intercept + slope * np.log10(ranks))
    axes.loglog(ranks, fitted, linestyle="--", linewidth=1.2,
                label=f"OLS slope = {slope:.3f}")

    axes.set_xlabel("rank")
    axes.set_ylabel("frequency")
    axes.set_title("Zipf's law - SC of India judgments")
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_ccdf(frequencies, path):
    """Complementary CDF of word frequencies - a cleaner view of the tail."""
    sorted_frequencies = np.sort(frequencies)
    survival = 1.0 - np.arange(len(sorted_frequencies)) / len(sorted_frequencies)

    figure, axes = plt.subplots(figsize=(7, 5))
    axes.loglog(sorted_frequencies, survival, linewidth=1.2)
    axes.set_xlabel("word frequency x")
    axes.set_ylabel("P(X >= x)")
    axes.set_title("Complementary CDF of word frequencies")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_heaps(xs, ys, path):
    """Heaps' law: vocabulary growth, with the fitted exponent in the legend."""
    log_xs = np.log10(xs)
    log_ys = np.log10(ys)
    beta, log_k = np.polyfit(log_xs, log_ys, 1)

    figure, axes = plt.subplots(figsize=(7, 5))
    axes.loglog(xs, ys, linewidth=1.2, label="observed")

    fitted = 10 ** (log_k + beta * log_xs)
    axes.loglog(xs, fitted, linestyle="--", linewidth=1.2,
                label=f"Heaps beta = {beta:.3f}")

    axes.set_xlabel("tokens seen")
    axes.set_ylabel("vocabulary size")
    axes.set_title("Heaps' law - vocabulary growth")
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

    return beta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["corpus", "metadata"], default="corpus")
    parser.add_argument("--corpus-dir", default="corpus")
    parser.add_argument("--metadata", default="sc_judgments_metadata.csv")
    parser.add_argument("--drop-boilerplate", action="store_true")
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    if args.source == "corpus":
        tokens = load_corpus_tokens(args.corpus_dir)
    else:
        tokens = load_metadata_tokens(args.metadata)

    table = build_rank_table(tokens, args.drop_boilerplate)

    total_tokens = len(tokens)
    vocabulary_size = len(table)
    hapax_count = sum(1 for _, _, count in table if count == 1)

    print()
    print(f"tokens          : {total_tokens}")
    print(f"vocabulary      : {vocabulary_size}")
    print(f"hapax legomena  : {hapax_count} "
          f"({100.0 * hapax_count / vocabulary_size:.1f}% of vocabulary)")
    print()
    print(f"top {args.top} words:")

    for rank, word, count in table[:args.top]:
        share = 100.0 * count / total_tokens
        print(f"  {rank:>4}  {word:<20} {count:>10}  {share:>6.3f}%")

    with open("rank_frequency.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "word", "frequency", "relative_frequency"])

        for rank, word, count in table:
            writer.writerow([rank, word, count, count / total_tokens])

    print()
    print("wrote rank_frequency.csv")

    ranks = np.array([row[0] for row in table], dtype=float)
    frequencies = np.array([row[2] for row in table], dtype=float)

    slope, intercept = fit_ols(ranks, frequencies)
    print(f"OLS log-log slope       : {slope:.4f}  (Zipf predicts about -1)")

    mle = fit_mle(frequencies)

    if mle is not None:
        alpha, x_min, ks_distance, ratio, p_value = mle
        print(f"MLE power-law alpha     : {alpha:.4f}")
        print(f"fitted x_min            : {x_min:.0f}")
        print(f"KS distance             : {ks_distance:.4f}")
        print(f"power-law vs lognormal  : LLR = {ratio:.2f}, p = {p_value:.4f}")
        print("  (LLR > 0 favours power law; p < 0.05 means the sign is meaningful)")

    plot_loglog(ranks, frequencies, slope, intercept, "zipf_loglog.png")
    plot_ccdf(frequencies, "zipf_ccdf.png")

    xs, ys = heaps_curve(tokens, sample_points=400)
    beta = plot_heaps(xs, ys, "heaps.png")

    print(f"Heaps' law beta         : {beta:.4f}  (English prose is usually 0.4-0.6)")
    print()
    print("wrote zipf_loglog.png, zipf_ccdf.png, heaps.png")


if __name__ == "__main__":
    main()
