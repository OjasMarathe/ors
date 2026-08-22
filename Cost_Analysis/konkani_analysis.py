"""
Konkani Language Support — Cost Estimation Model
Information Retrieval Course Project

Reproduces every number used in the presentation:
- corpus sizing across 3 quality tiers
- data cost using the given $1,000 / 100,000 words assumption
- corpus split by data type (fluency / speech / translation / instruction)
- infra + people cost estimates for Sarvam and Google Search
- final combined totals

Change any constant below and re-run to see how the totals update.
"""

# ---------------------------------------------------------------------------
# STEP 1: The core rate given in the assignment
# ---------------------------------------------------------------------------
COST_PER_UNIT = 1000       # dollars
WORDS_PER_UNIT = 100_000    # words
COST_PER_WORD = COST_PER_UNIT / WORDS_PER_UNIT  # $0.01/word


def data_cost(words: float) -> float:
    """Apply the $1,000 per 100,000 words formula to any word count."""
    return (words / WORDS_PER_UNIT) * COST_PER_UNIT


# ---------------------------------------------------------------------------
# STEP 2: Quality tiers (corpus size decision)
# ---------------------------------------------------------------------------
TIERS = {
    "Tier 1 - MVP":                 10_000_000,     # 10M words
    "Tier 2 - Production (chosen)": 500_000_000,    # 500M words
    "Tier 3 - Parity with Hindi":   5_000_000_000,  # 5B words
}

CHOSEN_TIER = "Tier 2 - Production (chosen)"
CHOSEN_WORDS = TIERS[CHOSEN_TIER]


# ---------------------------------------------------------------------------
# STEP 3: Split the chosen tier's corpus by data type/purpose
# (must sum to CHOSEN_WORDS)
# ---------------------------------------------------------------------------
CORPUS_SPLIT = {
    "General text (fluency)":        300_000_000,
    "Speech transcripts (voice)":    100_000_000,
    "Parallel/translation pairs":     50_000_000,
    "Instruction / RLHF-style data":  50_000_000,
}

assert sum(CORPUS_SPLIT.values()) == CHOSEN_WORDS, "Corpus split must sum to the chosen tier's word count"


# ---------------------------------------------------------------------------
# STEP 4: Split the corpus cost between the two companies
# (each gets the subset of words most relevant to their product)
# ---------------------------------------------------------------------------
COMPANY_WORDS = {
    "Sarvam (LLM-focused corpus)":    300_000_000,
    "Google Search (search-focused corpus)": 200_000_000,
}


# ---------------------------------------------------------------------------
# STEP 5: Infra / compute / people costs (NOT from the word formula —
# these are reference-class estimates, i.e. based on typical costs for
# comparable engineering work; see project write-up for reasoning)
# ---------------------------------------------------------------------------
SARVAM_INFRA = {
    "Tokenizer + embedding init for Konkani script": 50_000,
    "Continued pretraining compute (GPU hours)":     500_000,
    "Instruction tuning / RLHF pass":                150_000,
    "Voice model training (STT/TTS)":                200_000,
    "Evaluation & safety testing":                   100_000,
    "Inference/serving deployment infra":            150_000,
    "Engineering team (~10 people, 6 months)":       750_000,
}

GOOGLE_INFRA = {
    "Crawler/indexer updates":                        200_000,
    "Multilingual ranking/embedding model retraining": 500_000,
    "Spell-check & query-understanding models":        150_000,
    "Knowledge Graph / entity linking":                200_000,
    "Human search-quality raters":                     400_000,
    "Voice search integration (ASR/TTS)":              500_000,
    "Engineering team (~15 people, 9 months)":       2_025_000,
    "Localization/legal/compliance QA":                300_000,
}


# ---------------------------------------------------------------------------
# CALCULATIONS
# ---------------------------------------------------------------------------
def fmt(n):
    return f"${n:,.0f}"


def run_model():
    print("=" * 70)
    print("STEP 1 — Core rate")
    print("=" * 70)
    print(f"${COST_PER_UNIT} per {WORDS_PER_UNIT:,} words = ${COST_PER_WORD:.2f} / word\n")

    print("=" * 70)
    print("STEP 2 — Tier comparison (data cost only)")
    print("=" * 70)
    for name, words in TIERS.items():
        marker = "  <-- CHOSEN" if name == CHOSEN_TIER else ""
        print(f"{name:35s} {words:>15,} words   {fmt(data_cost(words)):>15}{marker}")
    print()

    print("=" * 70)
    print(f"STEP 3 — Corpus split for {CHOSEN_TIER} ({CHOSEN_WORDS:,} words)")
    print("=" * 70)
    for name, words in CORPUS_SPLIT.items():
        pct = words / CHOSEN_WORDS * 100
        print(f"{name:35s} {words:>15,} words   ({pct:4.0f}%)")
    print()

    print("=" * 70)
    print("STEP 4 — Data cost split by company")
    print("=" * 70)
    company_data_costs = {}
    for name, words in COMPANY_WORDS.items():
        cost = data_cost(words)
        company_data_costs[name] = cost
        print(f"{name:40s} {words:>15,} words   {fmt(cost):>15}")
    total_data_cost = sum(company_data_costs.values())
    print(f"{'TOTAL DATA COST':40s} {'':>25}{fmt(total_data_cost):>15}\n")

    print("=" * 70)
    print("STEP 5a — Sarvam infra/compute/people cost")
    print("=" * 70)
    for name, cost in SARVAM_INFRA.items():
        print(f"{name:45s} {fmt(cost):>15}")
    sarvam_infra_total = sum(SARVAM_INFRA.values())
    print(f"{'SARVAM INFRA SUBTOTAL':45s} {fmt(sarvam_infra_total):>15}\n")

    print("=" * 70)
    print("STEP 5b — Google Search infra/compute/people cost")
    print("=" * 70)
    for name, cost in GOOGLE_INFRA.items():
        print(f"{name:45s} {fmt(cost):>15}")
    google_infra_total = sum(GOOGLE_INFRA.values())
    print(f"{'GOOGLE INFRA SUBTOTAL':45s} {fmt(google_infra_total):>15}\n")

    print("=" * 70)
    print("FINAL TOTALS")
    print("=" * 70)
    sarvam_data = list(company_data_costs.values())[0]
    google_data = list(company_data_costs.values())[1]
    sarvam_total = sarvam_data + sarvam_infra_total
    google_total = google_data + google_infra_total
    combined_total = sarvam_total + google_total

    print(f"{'Company':25s}{'Data':>15}{'Infra/People':>18}{'Total':>15}")
    print(f"{'Sarvam':25s}{fmt(sarvam_data):>15}{fmt(sarvam_infra_total):>18}{fmt(sarvam_total):>15}")
    print(f"{'Google Search':25s}{fmt(google_data):>15}{fmt(google_infra_total):>18}{fmt(google_total):>15}")
    print(f"{'COMBINED':25s}{fmt(total_data_cost):>15}{fmt(sarvam_infra_total+google_infra_total):>18}{fmt(combined_total):>15}")


if __name__ == "__main__":
    run_model()
