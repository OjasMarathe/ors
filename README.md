# Zipf's Law across legal and religious corpora

Empirical testing of Zipf's law on six corpora in five scripts — Supreme Court of
India judgments (English), the Hebrew Bible, the Greek New Testament, the Qur'an
(Arabic), the Bhagavad Gita (Sanskrit) and the Pali Canon.

Coursework for an Information Retrieval course.

## Headline result

The fitted Zipf exponent depends heavily on **where you cut the tail**. Reporting a
single number without stating the fitting window is not meaningful.

| Corpus | Tokens | Types | Top 1,000 | Top 10,000 | All ranks |
|---|---|---|---|---|---|
| Pali Canon | 2,805,943 | 153,693 | −0.861 | **−1.001** | −1.291 |
| Hebrew Bible | 306,761 | 41,615 | −0.870 | −1.011 | −1.039 |
| Greek New Testament | 137,554 | 17,566 | −1.053 | −1.102 | −0.997 |
| Qur'an | 77,429 | 16,312 | −0.928 | −1.014 | −0.896 |
| Bhagavad Gita | 6,823 | 4,235 | −0.781 | — | −0.449 |
| SC judgments (English) | 291,026 | 9,609 | — | — | −1.456 |

The Pali Canon moves from −0.86 to −1.29 on the same data with the same estimator.
At 10,000 ranks it is at −1.001, essentially perfect Zipf. The whole-corpus figure is
dominated by the hapax shelf, not by the language.

## Full statistics

| Corpus | Tokens | Types | TTR | Hapax % | OLS slope | MLE α | KS |
|---|---|---|---|---|---|---|---|
| Pali Canon (whole) | 2,805,943 | 153,693 | 0.055 | 45.5% | −1.291 | 1.915 | 0.016 |
| — Vinaya | 422,916 | 32,405 | 0.077 | 40.3% | −1.229 | 2.008 | 0.010 |
| — Sutta | 1,595,681 | 127,027 | 0.080 | 47.8% | −1.199 | 2.104 | 0.009 |
| — Abhidhamma | 787,346 | 17,874 | 0.023 | 30.0% | −1.572 | 1.680 | 0.027 |
| Hebrew Bible (whole words) | 306,761 | 41,615 | 0.136 | 53.3% | −1.039 | 1.920 | 0.013 |
| Hebrew Bible (morphemes split) | 471,724 | 19,808 | 0.042 | 42.9% | −1.303 | 1.917 | 0.017 |
| Greek New Testament | 137,554 | 17,566 | 0.128 | 56.3% | −0.997 | 1.953 | 0.003 |
| Qur'an | 77,429 | 16,312 | 0.211 | 60.7% | −0.896 | 2.010 | 0.004 |
| Bhagavad Gita | 6,823 | 4,235 | 0.621 | 84.0% | −0.449 | 2.894 | 0.023 |
| SC judgment prose | 291,026 | 9,609 | 0.033 | 31.5% | −1.456 | — | — |
| SC case titles | 563,000 | 29,660 | 0.053 | 55.1% | −1.110 | 1.769 | 0.017 |

## Three findings worth writing up

**1. Morphology, not vocabulary, drives much of the tail.** The Westminster Leningrad
Codex marks prefix morphemes (conjunction, article, prepositions) that attach to the
following word in writing. Keeping words whole gives 41,615 types; splitting at those
boundaries gives 19,808 — the vocabulary halves and the exponent moves from −1.04 to
−1.30. See `02-scriptures/results/morphology_effect.png`.

**2. Genre changes the exponent within one language.** The Abhidhamma basket is larger
than the entire Hebrew Bible (787k tokens) yet uses only 17,874 distinct forms, a
type-token ratio of 0.023 — roughly a third of the Sutta basket's. Its slope (−1.572)
is the steepest measured here. The paṭṭhāna method enumerates conditional relations
exhaustively over a closed technical vocabulary, so a small term set recurs enormously
while the tail starves. One language, one transmission tradition, three genres,
measurably different exponents.

**3. Small corpora produce fake violations.** The Bhagavad Gita's −0.449 says nothing
about Sanskrit. At 6,823 tokens the curve runs out of data before a tail can form.
Compounding it, written Sanskrit fuses adjacent words (sandhi), which manufactures
one-off forms — hence 84% hapax and a TTR of 0.62 that no natural language has.

## Layout

```
01-supreme-court/      SC of India judgments (English)
  data/                43,495-judgment metadata index with direct PDF URLs
  scripts/             downloader, Zipf analysis, plotting
  corpus/              122 judgment documents used for the prose analysis
  results/             figures and rank tables

02-scriptures/         Hebrew, Greek, Arabic, Sanskrit
  scripts/             corpus builder + analysis
  corpus/              tokenised text, one file per corpus
  results/             figures, summary table

03-tipitaka/           Pali Canon, split by basket
04-distribution-plots/ full-vocabulary and windowed plots for every corpus
rank-tables/           complete rank/frequency/relative-frequency tables (large ones gzipped)
```

Each sub-directory has its own README with the source edition, licence, and the exact
preprocessing decisions used.

## Data sources and licences

| Corpus | Source | Licence |
|---|---|---|
| SC judgments | AWS Open Data `indian-supreme-court-judgments`; index via `darshjme/india-supreme-court-search` | CC-BY-4.0; judgments exempt under s.52(1)(q)(iv), Copyright Act 1957 |
| Judgment prose sample | MILDSum (Law-AI, IIT Kharagpur); ILDC/CJPE (Exploration Lab, IIT Kanpur) | research use; CC-BY-NC 4.0 |
| Hebrew Bible | Westminster Leningrad Codex, `openscriptures/morphhb` | CC-BY 4.0 |
| Greek NT | SBLGNT / MorphGNT, `morphgnt/sblgnt` | CC-BY-SA 3.0 + SBLGNT licence |
| Qur'an | Arabic text, `risan/quran-json` | MIT (code); text public domain |
| Bhagavad Gita | `gita/gita` | MIT |
| Pali Canon | Mahāsaṅgīti edition, `suttacentral/bilara-data` | CC0 |

Cite the digital editions, not just the texts. Papers to cite: Malik et al. (ACL 2021,
ILDC); Datta et al. (EMNLP 2023, MILDSum).

## Reproducing

```
pip install -r requirements.txt

python 02-scriptures/scripts/build_scripture_corpora.py
python 02-scriptures/scripts/analyse_scriptures.py
python 03-tipitaka/scripts/analyse_tipitaka.py
python 04-distribution-plots/scripts/plot_all_words.py
python 04-distribution-plots/scripts/plot_windows.py
```

Corpus building needs these repositories cloned into `scripture_src/`:

```
git clone --depth 1 https://github.com/openscriptures/morphhb.git
git clone --depth 1 https://github.com/morphgnt/sblgnt.git
git clone --depth 1 https://github.com/risan/quran-json.git
git clone --depth 1 https://github.com/gita/gita.git
git clone --depth 1 https://github.com/suttacentral/bilara-data.git
```

Rendering Hebrew, Arabic and Devanagari bar charts needs the fonts
`fonts-sil-ezra`, `fonts-hosny-amiri` and `fonts-noto-core`.

## Known limitations

- The judgment-prose corpus (122 documents) is annotator-selected passages, not a
  uniform random sample. Rebuild it from the official PDFs with
  `01-supreme-court/scripts/download_judgments.py` before quoting those numbers.
- Sandhi is not segmented for Sanskrit or Pali, so type counts are inflated in both.
  Within-Pali basket comparisons are unaffected since the bias applies equally.
- Corpus sizes vary 400× across the set. Zipf exponents are size-sensitive; subsample
  to a common token count before comparing across corpora.
- OLS on log-log is reported throughout for comparability, but the MLE (α) column is
  the defensible estimate.
