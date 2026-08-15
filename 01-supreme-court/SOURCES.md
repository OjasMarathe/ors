# Where the judgment-prose corpus came from

The 122 documents in `judgment_corpus/` were assembled from two open research
datasets, because the full AWS judgment-PDF bucket was not reachable from the
machine that produced these plots.

| Source | Contribution | Licence |
|---|---|---|
| **MILDSum** (Law-AI, IIT Kharagpur) — `github.com/Law-AI/MILDSum` | 10 complete judgment texts | research use, cite the paper |
| **ILDC / CJPE** (Exploration Lab, IIT Kanpur) — `github.com/Exploration-Lab/CJPE` | 112 documents of expert-annotated judgment passages | CC-BY-NC 4.0 |

Cite both if you use these numbers:

- Malik et al., *ILDC for CJPE: Indian Legal Documents Corpus for Court Judgment
  Prediction and Explanation*, ACL-IJCNLP 2021.
- Datta et al., *MILDSum: A Novel Benchmark Dataset for Multilingual Summarization
  of Indian Legal Case Judgments*, EMNLP 2023.

**Important caveat for your report:** the ILDC portion consists of *passages
selected by annotators*, not whole judgments, so it is not a uniform random
sample of judgment text. It is fine for demonstrating the shape of the
distribution, but the numbers below should be regenerated on a proper random
sample before they go in a final submission. Use `download_judgments.py` with
`sc_judgments_metadata.csv` to build that sample from the official PDFs.

# Results in this folder

| Corpus | Documents | Tokens | Vocabulary | Hapax | OLS slope |
|---|---|---|---|---|---|
| Judgment prose | 122 | 291,026 | 9,609 | 31.5% | -1.456 |
| Case titles | 43,495 | 563,000 | 29,660 | 55.1% | -1.110 |
