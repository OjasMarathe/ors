# Zipf's Law on Supreme Court of India Judgments

A ready-to-run pipeline for empirically testing Zipf's law on judgments of the
Supreme Court of India, 1950–2026.

## What's in here

| File | What it is |
|---|---|
| `sc_judgments_metadata.csv` | 43,495 SC judgments: title, parties, judge, citation, neutral citation, CNR, decision date, disposal, available languages, and a **direct PDF URL** for each |
| `download_judgments.py` | Downloads a stratified sample of judgment PDFs and extracts plain text into `corpus/` |
| `zipf_analysis.py` | Rank–frequency analysis, OLS + MLE power-law fits, Heaps' law, plots |
| `demo_metadata_run/` | Output from a test run on case titles only, so you can see the shape of the results before downloading anything |

## Source and licence

The judgments come from the **Indian Supreme Court Judgments** open dataset on the
AWS Registry of Open Data (`s3://indian-supreme-court-judgments/`), derived from the
official eCourts/SCI record. It is licensed **CC-BY-4.0** — cite it in your report.
The metadata CSV here was rebuilt from the index published by the
`darshjme/india-supreme-court-search` project, which draws from the same bucket.

Separately, judgments of Indian courts are exempt from copyright under
s.52(1)(q)(iv) of the Copyright Act, 1957, so the text itself is free to use.

## Running it

```
pip install requests pypdf matplotlib numpy powerlaw

python download_judgments.py --per-year 30 --start-year 1990 --end-year 2025
python zipf_analysis.py --source corpus
```

`--per-year 30` over 36 years is about 1,080 judgments, which is roughly 3–6 million
tokens — more than enough for a clean Zipf curve, and it downloads in a few minutes.
Scale up if you want.

If you want to see something running right now without downloading anything:

```
python zipf_analysis.py --source metadata
```

That tokenises case titles and judge names instead of judgment bodies.

## Two things to watch out for

**Pre-1970 PDFs are often scanned images.** `pypdf` will return almost nothing for
them. The downloader silently drops any file yielding under 1,500 characters and
reports the count at the end. If you want the early decades you'll need OCR
(`ocrmypdf`), which is a real time cost — I'd start from 1990 and treat older years
as an optional extension.

**Header/footer boilerplate pollutes the head of the distribution.** Words like
"REPORTABLE" and the digital-signature block appear once or more per document, which
puts them at an artificially high rank. Run the analysis both with and without
`--drop-boilerplate` and show both — the difference is a legitimate finding about
corpus hygiene, not a bug to hide.

## Why judgments are a good choice for this

A generic English corpus gives you a straight line and not much to say. Legal prose
is interesting because it deviates in specific, explainable ways:

- **A fat, distorted head.** Function words still dominate, but domain terms
  (*appellant*, *respondent*, *section*, *court*, *held*, *learned*) push into ranks
  that ordinary prose reserves for common vocabulary.
- **Formulaic repetition.** Judgments quote statutes and prior judgments verbatim,
  which inflates mid-frequency terms and tends to steepen the fitted exponent
  relative to conversational or journalistic text.
- **A strange tail.** Party names, place names, and transliterated Indian names are
  overwhelmingly hapax legomena, so the vocabulary keeps growing long after a normal
  corpus would saturate. Expect a high Heaps' exponent.
- **It changes over time.** Split the corpus by decade and fit separately —
  1950s judgments are shorter and more Latinate, post-2000 judgments are longer and
  more procedural. A shifting exponent across decades is a genuinely nice result and
  almost nobody submits it.

## Methodology note for the report

Do **not** report only the OLS slope of the log–log plot. It is the standard textbook
method and it is biased, because the noisy low-frequency tail contains most of the
points and dominates the least-squares objective. `zipf_analysis.py` reports it for
comparison, but the number to lead with is the Clauset–Shalizi–Newman maximum
likelihood estimate of α with a fitted `x_min`, plus the KS goodness-of-fit distance
and the power-law vs. lognormal likelihood ratio. Showing that you know the
difference is the easiest way to separate your submission from the rest of the class.

Also plot the complementary CDF alongside the rank–frequency curve. It uses every
data point without binning and makes tail deviations far easier to see.

## Suggested structure for the writeup

1. Corpus construction and cleaning decisions (with token/vocabulary counts)
2. Rank–frequency curve, OLS fit, and why you are not stopping there
3. MLE fit: α, `x_min`, KS distance, distribution comparison
4. Where the law breaks — the head, the boilerplate, the name-heavy tail
5. Heaps' law and the consistency check against α
6. Decade-wise comparison (optional, and the part most likely to earn extra credit)
