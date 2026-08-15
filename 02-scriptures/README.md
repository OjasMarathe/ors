# Zipf's Law across scriptures in their original languages

Hebrew Bible, Greek New Testament, Qur'an (Arabic) and Bhagavad Gita (Sanskrit),
tokenised in their original scripts and fitted against Zipf's law.

## Sources (all cloned from public GitHub repositories)

| Corpus | Edition | Repository | Licence |
|---|---|---|---|
| Hebrew Bible | Westminster Leningrad Codex (OSHB) | `openscriptures/morphhb` | CC-BY 4.0 |
| Greek New Testament | SBLGNT / MorphGNT | `morphgnt/sblgnt` | CC-BY-SA 3.0 (MorphGNT), SBLGNT licence |
| Qur'an | Arabic text, Tanzil-derived | `risan/quran-json` | MIT (code), text is public domain |
| Bhagavad Gita | Devanagari verse text | `gita/gita` | MIT |

The underlying texts are ancient and in the public domain; the licences above cover
the specific digital editions. Cite the editions, not just "the Bible".

## Preprocessing decisions (state these explicitly in your report)

- **Hebrew** — cantillation marks and vowel points (U+0591–U+05C7) stripped, leaving
  consonantal text. The source marks prefix morphemes with `/`; the main corpus
  keeps orthographic words whole, and `hebrew_ot_morphemes` splits at those
  boundaries.
- **Greek** — the MorphGNT *normalised* word-form column, which folds sentence-initial
  capitals and final-position accent variation.
- **Arabic** — tashkeel, Quranic annotation signs and tatweel stripped. Letter forms
  (alef variants, ta marbuta) left as written.
- **Sanskrit** — dandas, verse numbers and avagraha removed; whitespace tokenisation.
  **Sandhi is not undone** — see the caveat below.

## Results

| Corpus | Tokens | Types | TTR | Hapax % | OLS slope | MLE α | KS |
|---|---|---|---|---|---|---|---|
| Hebrew Bible (whole words) | 306,761 | 41,615 | 0.136 | 53.3% | −1.039 | 1.920 | 0.013 |
| Greek New Testament | 137,554 | 17,566 | 0.128 | 56.3% | −0.997 | 1.953 | 0.003 |
| Qur'an | 77,429 | 16,312 | 0.211 | 60.7% | −0.896 | 2.010 | 0.004 |
| Bhagavad Gita | 6,823 | 4,235 | 0.621 | 84.0% | −0.449 | 2.894 | 0.023 |
| Hebrew (morphemes split) | 471,724 | 19,808 | 0.042 | 42.9% | −1.303 | 1.917 | 0.017 |

For reference, the SC of India judgment corpus gave an OLS slope of −1.456.

## Two caveats that must go in the writeup

**The Gita is too small.** 6,823 tokens is roughly a long magazine article. At that
size a Zipf fit is dominated by sampling noise, and the −0.45 slope is largely an
artifact of the curve being truncated — the corpus simply runs out of words before
the tail can develop. Report it as a demonstration of the small-sample failure mode,
not as a property of Sanskrit.

**Sandhi inflates the Sanskrit vocabulary.** Written Sanskrit verse fuses adjacent
words at their boundaries, so whitespace tokenisation produces forms that occur once
and never again. 84% hapax is mostly this. Splitting sandhi properly needs a
segmenter (`sanskrit_parser`, or the Heritage segmenter); doing that and re-running
would be the single strongest extension to this assignment.

## Reproducing

```
pip install matplotlib numpy powerlaw arabic-reshaper python-bidi
python build_scripture_corpora.py
python analyse_scriptures.py
```

Rendering the bar charts needs fonts for each script:
`fonts-sil-ezra` (Hebrew), `fonts-hosny-amiri` (Arabic), `fonts-noto-core` (Devanagari).
