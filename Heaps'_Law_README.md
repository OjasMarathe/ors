# Heaps' Law on Japanese anime subtitles, 1983–2024

V = K · N^β fitted across 3-year time bins, on morphologically segmented Japanese.

## Corpus

| | |
|---|---|
| Shows | 340 (of 346; 98.3% matched to an air year) |
| Episodes | 6,854 |
| Subtitle lines | 475,038 |
| Tokens (Japanese, segmented) | **3,273,982** |
| Distinct word forms | 57,215 |
| Year span | 1983–2024 |

**Sources.** Subtitles: `harikc456/anime-subs-mapping`. Air years: `manami-project/anime-offline-database`
(tag 2025-18), matched on title plus synonyms after ASCII-folding, with season
suffixes (`_S2`, `_Eng`) stripped on retry. The 6 unmatched shows are listed in
`corpus/unmatched.txt`.

**Segmentation.** Japanese has no whitespace, so tokens come from `fugashi` +
`unidic-lite` (MeCab/UniDic). Surface forms, not lemmas. Punctuation, symbols and
whitespace POS classes (補助記号, 空白, 記号) are dropped.

Subtitles are derivative works — cite the dataset, don't redistribute the raw text.

## Method

- **Shuffling.** Vocabulary growth depends on document order, so each curve is the
  mean of **30 random shuffles**, with a ±1 s.d. band. Unshuffled chronological
  order would bias β upward.
- **Shuffle unit.** Pseudo-documents of 500 tokens. Mean real episode length is 477
  tokens, so these approximate episode granularity while giving ~6,500 units.
- **Burn-in.** Heaps is not a power law at small N. Everything below **N = 10,000**
  is excluded from the fit and shaded grey on the log-log plot.
- **Thin bins.** Bins under 100,000 tokens can't support a fit. Pre-2004 bins are
  merged into one group; the 2022–2024 bin (one show) is folded into 2019–2021.

## Results

| Bin | Tokens | Types | β | K | R² | Zipf α | 1/α |
|---|---|---|---|---|---|---|---|
| 1983–2003 (merged) | 336,161 | 17,480 | 0.579 | 11.8 | 0.9969 | 1.276 | 0.784 |
| 2004–2006 | 452,529 | 22,042 | 0.594 | 10.3 | 0.9977 | 1.268 | 0.789 |
| 2007–2009 | 378,614 | 18,090 | 0.576 | 11.9 | 0.9967 | 1.293 | 0.773 |
| 2010–2012 | 714,462 | 26,355 | 0.572 | 13.0 | 0.9956 | 1.348 | 0.742 |
| 2013–2015 | 650,001 | 25,781 | 0.576 | 12.8 | 0.9957 | 1.326 | 0.754 |
| 2016–2018 | 559,059 | 23,887 | 0.581 | 12.0 | 0.9959 | 1.305 | 0.766 |
| 2019–2024 | 183,156 | 12,099 | 0.585 | 10.5 | 0.9981 | 1.214 | 0.824 |
| **POOLED** | **3,273,982** | **57,215** | **0.554** | **16.9** | **0.9924** | 1.511 | 0.662 |

## Reading the figures

`heaps_loglog.png` — the fit itself. Straight lines above the burn-in, R² > 0.995
everywhere. The shaded region is what gets excluded and why.

`heaps_linear.png` — the slowdown. The pooled curve reaches 30,000 types at 1M
tokens and only 57,215 at 3.27M: tripling the corpus adds under twice the
vocabulary. It never flattens, but the curvature is unmistakable.

`heaps_discovery_rate.png` — **the strongest figure**. New distinct words per 1,000
tokens, plotted against N. It falls from ~250 at N=1,000 to ~7 at N=3M — a 35×
decline, straight on log-log. This is the quantitative version of "the plateau":
β < 1 *is* a power-law decay in the discovery rate, and here you can read the
exponent straight off the slope.

`heaps_beta_over_time.png` — β and K per bin.

## What the numbers say

**β is remarkably stable across four decades**: 0.572 to 0.594, a spread of 0.022
across seven bins spanning 1983–2024. Anime dialogue has not become measurably more
or less lexically open since the 1980s. Given that the bins differ in size by 4×,
that stability is a real result rather than an artifact.

**The pooled β (0.554) is lower than every individual bin.** That is not a mistake —
it's the expected consequence of pooling. Bins share a large common core vocabulary,
so merging them adds tokens much faster than it adds types, which flattens the curve.
Worth a paragraph: it shows β is a property of a corpus, not of a language.

**K moves inversely to β** (K = 16.9 pooled versus 10–13 per bin). K and β trade off
in the fit — a higher intercept is compensated by a shallower slope. Never compare K
values across corpora without holding β roughly fixed.

**The β ≈ 1/α check partly holds.** Per-bin, 1/α lands at 0.74–0.82 against measured
β of 0.57–0.59: same ballpark, consistently 25–35% high. The relation is asymptotic
and assumes a pure power law over the whole range, which the Zipf curve isn't
(see the earlier Zipf work — α depends heavily on the fitting window). Reporting the
gap honestly, with that explanation, is better than pretending it matched.

## Reproducing

```
pip install fugashi unidic-lite pandas matplotlib numpy
git clone --depth 1 https://github.com/harikc456/anime-subs-mapping.git
# anime-offline-database JSON: raw.githubusercontent.com, tag 2025-18, minified

python scripts/build_anime_corpus.py    # writes heaps_corpus/, ~15 min
python scripts/analyse_heaps.py         # writes plots + heaps_parameters.csv
```
