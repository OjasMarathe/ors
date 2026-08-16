# Heaps' Law on Japanese anime subtitles

Vocabulary growth V(N) measured on a pooled corpus of morphologically segmented
Japanese, and the exponent measured as the local slope of that curve.

## Headline

**A single Heaps exponent does not describe this corpus.** The local slope
d(log V)/d(log N) declines continuously from 0.98 at N=4 to 0.40 at N=3.27M, with
no plateau anywhere. Any quoted beta is therefore a statement about a chosen
fitting range, not about the corpus.

| | |
|---|---|
| Shows | 340 (of 346; 98.3% matched to an air year) |
| Episodes | 6,854 |
| Subtitle lines | 475,038 |
| **Tokens (N)** | **3,273,982** |
| **Distinct word forms (V)** | **57,215** |
| Year span | 1983-2024 |

Local slope at selected corpus sizes:

| N | local slope |
|---|---|
| 1,000 | 0.784 |
| 10,000 | 0.712 |
| 100,000 | 0.595 |
| 1,000,000 | 0.441 |
| 3,000,000 | 0.398 |

If a single number is required: over N in [10^4, 3.27x10^6], V = 16.9 * N^0.554
(R^2 = 0.992) — quoted **with** its fitting range, and with the note above.

## A note on the figures

No fitted power law is drawn on any figure. A fitted line represents one arbitrary
choice of N_min, and the local-slope panel shows that no such choice is defensible;
drawing it would give one option a status the data does not support. The figures
show the measured curve and let the slope decline speak for itself.

## How β and K were obtained

Heaps' law is V = K · N^β. Taking logs gives

    log V = log K + β · log N

which is a straight line: **slope = β, intercept = log K**. So the fit is ordinary
least squares on the log-log points. `np.polyfit` returns slope 0.5538 and intercept
1.2267, hence β = 0.554 and K = 10^1.2267 = 16.9.

The points being fitted come from the vocabulary curve: shuffle the pseudo-documents,
walk the token stream keeping a running set of distinct forms, and record (N, V) at
400 log-spaced checkpoints. Repeat 30 times with different shuffles and average.

## Method

- **Shuffling.** Vocabulary growth depends on document order, so each curve is the
  mean of **30 random shuffles**, with a ±1 s.d. band. Chronological order would bias
  β upward.
- **Shuffle unit.** Pseudo-documents of 500 tokens (6,713 of them). Mean real episode
  length is 477 tokens, so these approximate episode granularity.
- **Burn-in.** Heaps is not a power law at small N. Everything below **N = 10,000** is
  excluded from the fit and shaded grey on the log-log plot.

## The fit is good but not perfect — say so in the report

A single power law slightly overshoots at the far end:

| N | fitted V | actual V |
|---|---|---|
| 3,273,982 | 68,357 | **57,215** (−19.5%) |

The real curve bends downward relative to a pure power law, so β depends on the
fitting window — the same problem as the Zipf exponent:

| Fitting window | β | K | R² |
|---|---|---|---|
| 1 – all (no burn-in) | 0.7045 | 2.75 | 0.9918 |
| **10,000 – all (headline)** | **0.5538** | **16.85** | **0.9924** |
| 10,000 – 100,000 | 0.6638 | 5.27 | 0.9995 |
| 100,000 – 1,000,000 | 0.5178 | 28.48 | 0.9985 |
| 1,000,000 – all | 0.4057 | 130.90 | 0.9996 |

β falls from 0.70 to 0.41 depending on where you fit. Note also that **K moves
inversely to β** — they trade off against each other in the fit, so K is not
independently interpretable and should never be compared across corpora without
holding β roughly fixed.

Reporting the window dependence is stronger than quoting one number as if it were
the answer.

## The figures

`heaps_loglog.png` — the fit. Straight above the burn-in; the shaded region shows
what was excluded and why. You can see the slight downward bend past N ≈ 10^6.

`heaps_linear.png` — the slowdown. 1M tokens → 35,332 types; 2M → 47,252; 3M →
55,367. The second million adds 11,920 types, the third only 8,115. It never flattens,
but the curvature is unmistakable.

`heaps_discovery_rate.png` — **the strongest figure.** New distinct words per 1,000
tokens against N: **340 at N=1,000 falling to 7 at N=3.27M, a 51× decline**, close to
straight on log-log. This is the quantitative version of "the plateau": β < 1 *is* a
power-law decay in discovery rate, with slope β−1 = −0.446. The observed curve is
slightly steeper than that prediction at the far end, which is the same bend visible
in the log-log plot.

## Corpus

**Sources.** Subtitles: `harikc456/anime-subs-mapping`. Air years:
`manami-project/anime-offline-database` (tag 2025-18), matched on title plus synonyms
after ASCII-folding, with season suffixes (`_S2`, `_Eng`) stripped on retry. The 6
unmatched shows are listed in `corpus/unmatched.txt`.

**Segmentation.** Japanese has no whitespace, so tokens come from `fugashi` +
`unidic-lite` (MeCab/UniDic). Surface forms, not lemmas. Punctuation, symbol and
whitespace POS classes (補助記号, 空白, 記号) are dropped.

β is sensitive to segmenter choice — UniDic splits fairly aggressively, so a different
tokenizer would shift the absolute value. State which one you used.

Subtitles are derivative works. Cite the dataset; don't redistribute the raw text.

## Reproducing

```
pip install fugashi unidic-lite pandas matplotlib numpy
git clone --depth 1 https://github.com/harikc456/anime-subs-mapping.git
# anime-offline-database JSON: raw.githubusercontent.com, tag 2025-18, minified

python scripts/build_anime_corpus.py    # writes heaps_corpus/, ~15 min
python scripts/analyse_heaps.py         # writes plots + parameter tables
```

`build_anime_corpus.py` still writes the corpus into 3-year folders — that is only a
directory layout, and the analysis pools everything regardless.
