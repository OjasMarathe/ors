# Zipf's Law on the Pali Canon (Tipitaka)

The same pipeline applied to the Bible, Qur'an and Bhagavad Gita, now run on the
Theravada Pali Canon in romanised Pali — and split by basket.

## Source

**SuttaCentral `bilara-data`**, path `root/pli/ms/` — the **Mahasangiti** edition of
the Tipitaka (the text approved at the Sixth Buddhist Council, 1954–56), segmented
into addressable units. Repository: `github.com/suttacentral/bilara-data`.

The Mahasangiti root text is released into the **public domain (CC0)** by
SuttaCentral. Cite the edition and SuttaCentral as the digital source.

Coverage: 7,288 files across all three baskets — Vinaya (422), Sutta (5,764),
Abhidhamma (1,102).

## Preprocessing

- Romanised Pali, lowercased; tokens matched as Latin letters plus the standard
  Pali diacritics (ā ī ū ṁ ṃ ṅ ñ ṭ ḍ ṇ ḷ).
- **Inline HTML stripped.** Segments carry `<b>` tags around lemma citations; left
  in, the tag letter is tokenised as a word and lands at rank 8 with ~17,700
  occurrences. This is worth a sentence in the report as a cleaning finding.
- **Peyyala markers removed.** `pe` and `la` are editorial abbreviations standing in
  for repeated passages, not Pali words.
- Sandhi and compounds are **not** segmented — same caveat as the Gita.

## Results

| Corpus | Tokens | Types | TTR | Hapax % | OLS slope | MLE α | KS |
|---|---|---|---|---|---|---|---|
| Whole canon | 2,805,943 | 153,693 | 0.055 | 45.5% | −1.291 | 1.915 | 0.016 |
| Vinaya (monastic rules) | 422,916 | 32,405 | 0.077 | 40.3% | −1.229 | 2.008 | 0.010 |
| Sutta (discourses) | 1,595,681 | 127,027 | 0.080 | 47.8% | −1.199 | 2.104 | 0.009 |
| Abhidhamma (systematic analysis) | 787,346 | 17,874 | 0.023 | 30.0% | −1.572 | 1.680 | 0.027 |

## Cross-corpus comparison (all work so far)

| Corpus | Tokens | TTR | OLS slope |
|---|---|---|---|
| Pali Canon | 2,805,943 | 0.055 | −1.291 |
| Hebrew Bible | 306,761 | 0.136 | −1.039 |
| Greek New Testament | 137,554 | 0.128 | −0.997 |
| Qur'an | 77,429 | 0.211 | −0.896 |
| Bhagavad Gita | 6,823 | 0.621 | −0.449 |
| SC of India judgments (English) | 291,026 | 0.033 | −1.456 |

## The finding worth building the report around

The **Abhidhamma is a genuine outlier, and it is not a size artifact**. At 787,346
tokens it is larger than the Hebrew Bible, yet it uses only 17,874 distinct word
forms — a type-token ratio of 0.023, roughly a third of the Sutta basket's. Its
slope (−1.572) is the steepest of anything measured here, steeper even than the
Supreme Court judgments.

That is what extreme formulaic repetition looks like in a Zipf curve. The Abhidhamma
works through permutations of a fixed technical vocabulary — the *paṭṭhāna* method
enumerates conditional relations exhaustively — so a small closed set of terms
recurs at enormous frequency while the tail is starved.

Look at the basket plot: the Abhidhamma curve sits **above** the others through the
head and mid-range, then **falls below** them past rank ~1,000 and terminates early.
Compare that against the Sutta basket, which is ordinary narrative and doctrinal
prose and tracks much closer to the ideal slope.

This gives you a controlled comparison that is hard to get elsewhere: one language,
one register tradition, one transmission history, three genres — and a measurable
difference in exponent that has a documented textual explanation behind it.

## Reproducing

```
git clone --depth 1 https://github.com/suttacentral/bilara-data.git
pip install matplotlib numpy powerlaw
python analyse_tipitaka.py
```
