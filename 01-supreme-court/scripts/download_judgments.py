"""
Download a sample of Supreme Court of India judgment PDFs and extract their text.

The PDFs live in a public AWS Open Data bucket (no credentials needed, plain HTTPS).
This script reads sc_judgments_metadata.csv, picks a stratified sample across years,
downloads the PDFs, extracts text, and writes one .txt file per judgment into corpus/.

Usage:
    pip install requests pypdf
    python download_judgments.py --per-year 25 --start-year 1990 --end-year 2025

Output:
    pdfs/    raw downloaded PDFs
    corpus/  extracted plain text, one file per judgment
    corpus_manifest.csv  which judgment produced which text file
"""

import argparse
import csv
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import requests
from pypdf import PdfReader


METADATA_FILE = "sc_judgments_metadata.csv"
PDF_DIR = "pdfs"
TEXT_DIR = "corpus"
MANIFEST_FILE = "corpus_manifest.csv"

# A scanned 1950s judgment often yields almost no extractable text.
# Anything below this character count is treated as an extraction failure.
MIN_CHARS = 1500


def load_metadata(path):
    """Read the metadata CSV into a list of dicts."""
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def stratified_sample(rows, per_year, start_year, end_year, seed):
    """Pick up to `per_year` judgments from each year in the range."""
    by_year = defaultdict(list)

    for row in rows:
        try:
            year = int(row["year"])
        except (ValueError, TypeError):
            continue

        if year < start_year:
            continue
        if year > end_year:
            continue
        if not row.get("pdf_url"):
            continue

        by_year[year].append(row)

    rng = random.Random(seed)
    sample = []

    for year in sorted(by_year):
        candidates = by_year[year]
        rng.shuffle(candidates)
        sample.extend(candidates[:per_year])

    return sample


def safe_name(row):
    """Build a filesystem-safe base name for a judgment."""
    base = row["pdf_url"].rsplit("/", 1)[-1]
    base = base.replace(".pdf", "")
    base = "".join(ch for ch in base if ch.isalnum() or ch in "._-")
    return f"{row['year']}_{row['id']}_{base}"


def download_one(row, session):
    """Download a single PDF. Returns (row, pdf_path) or (row, None) on failure."""
    name = safe_name(row)
    pdf_path = os.path.join(PDF_DIR, name + ".pdf")

    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
        return row, pdf_path

    try:
        response = session.get(row["pdf_url"], timeout=60)
    except requests.RequestException as error:
        print(f"  network error {name}: {error}", file=sys.stderr)
        return row, None

    if response.status_code != 200:
        print(f"  http {response.status_code} {name}", file=sys.stderr)
        return row, None

    with open(pdf_path, "wb") as handle:
        handle.write(response.content)

    return row, pdf_path


def extract_text(pdf_path):
    """Pull plain text out of a PDF. Returns None if extraction is too thin."""
    try:
        reader = PdfReader(pdf_path)
    except Exception as error:
        print(f"  unreadable pdf {pdf_path}: {error}", file=sys.stderr)
        return None

    chunks = []

    for page in reader.pages:
        try:
            page_text = page.extract_text()
        except Exception:
            page_text = None

        if page_text:
            chunks.append(page_text)

    text = "\n".join(chunks)

    if len(text) < MIN_CHARS:
        return None

    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-year", type=int, default=25)
    parser.add_argument("--start-year", type=int, default=1990)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata", default=METADATA_FILE)
    args = parser.parse_args()

    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(TEXT_DIR, exist_ok=True)

    print(f"reading {args.metadata} ...")
    rows = load_metadata(args.metadata)
    print(f"  {len(rows)} judgments in metadata")

    sample = stratified_sample(
        rows,
        args.per_year,
        args.start_year,
        args.end_year,
        args.seed,
    )
    print(f"  sampled {len(sample)} judgments "
          f"({args.start_year}-{args.end_year}, up to {args.per_year}/year)")

    session = requests.Session()
    session.headers.update({"User-Agent": "zipf-coursework/1.0"})

    downloaded = []
    started = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []

        for row in sample:
            futures.append(pool.submit(download_one, row, session))

        for index, future in enumerate(as_completed(futures), start=1):
            row, pdf_path = future.result()

            if pdf_path is not None:
                downloaded.append((row, pdf_path))

            if index % 25 == 0:
                elapsed = time.time() - started
                print(f"  downloaded {index}/{len(sample)} ({elapsed:.0f}s)")

    print(f"downloaded {len(downloaded)} PDFs")
    print("extracting text ...")

    manifest = []
    failures = 0

    for row, pdf_path in downloaded:
        name = safe_name(row)
        text_path = os.path.join(TEXT_DIR, name + ".txt")

        if os.path.exists(text_path):
            manifest.append((row, text_path))
            continue

        text = extract_text(pdf_path)

        if text is None:
            failures += 1
            continue

        with open(text_path, "w", encoding="utf-8") as handle:
            handle.write(text)

        manifest.append((row, text_path))

    print(f"  extracted {len(manifest)} text files, {failures} unusable "
          f"(likely scanned images)")

    with open(MANIFEST_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "year", "title", "judge", "disposal", "text_file"])

        for row, text_path in manifest:
            writer.writerow([
                row["id"],
                row["year"],
                row["title"],
                row["judge"],
                row["disposal"],
                text_path,
            ])

    print(f"wrote {MANIFEST_FILE}")
    print(f"corpus is in ./{TEXT_DIR}/ - now run: python zipf_analysis.py")


if __name__ == "__main__":
    main()
