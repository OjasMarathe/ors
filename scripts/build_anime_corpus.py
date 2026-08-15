"""
Build a Japanese anime subtitle corpus, binned into 3-year groups.

Sources:
  harikc456/anime-subs-mapping        475k aligned JP/EN subtitle lines, 347 shows
  manami-project/anime-offline-database   title -> air year, plus synonyms

Output:
  heaps_corpus/<bin>/<show>.txt   one file per show, whitespace-joined JP tokens
  heaps_corpus/manifest.csv       show, year, bin, episodes, tokens
  heaps_corpus/unmatched.txt      shows with no year match
"""

import csv
import json
import os
import re
import unicodedata
from collections import defaultdict

import fugashi
import pandas as pd


SUBS_CSV = "heaps_src/harikc456_anime-subs-mapping/clean_jp-en_dataset.csv"
DB_JSON = "heaps_src/manami_db.json"
OUT_DIR = "heaps_corpus"

BIN_WIDTH = 3

# Punctuation, symbols and whitespace as tagged by UniDic. These are not words.
DROP_POS = {"補助記号", "空白", "記号"}

# Season and language suffixes that the subtitle filenames carry but the
# anime database titles do not.
SUFFIX = re.compile(
    r"(_s\d+|_season\s*\d+|_eng|_english|_sub|_dub|_bd|_tv)$",
    re.IGNORECASE,
)


def normalise(text):
    """Fold a title down to bare lowercase alphanumerics for matching."""
    folded = unicodedata.normalize("NFKD", str(text))
    folded = folded.encode("ascii", "ignore").decode()
    folded = folded.lower()
    folded = re.sub(r"[^a-z0-9]+", " ", folded).strip()
    return folded


def build_year_index():
    """Map every known title and synonym to its air year."""
    with open(DB_JSON, encoding="utf-8") as handle:
        database = json.load(handle)

    entries = database["data"] if isinstance(database, dict) else database
    index = {}

    for entry in entries:
        season = entry.get("animeSeason") or {}
        year = season.get("year")

        if not year:
            continue

        names = [entry.get("title", "")]
        names.extend(entry.get("synonyms") or [])

        for name in names:
            key = normalise(name)

            if key and key not in index:
                index[key] = year

    return index


def lookup_year(show_name, index):
    """Try the raw name, then progressively strip season/language suffixes."""
    candidate = show_name.replace("_", " ")
    key = normalise(candidate)

    if key in index:
        return index[key]

    stripped = show_name

    for _ in range(3):
        new = SUFFIX.sub("", stripped)

        if new == stripped:
            break

        stripped = new
        key = normalise(stripped.replace("_", " "))

        if key in index:
            return index[key]

    # Last resort: drop a trailing standalone number or roman numeral.
    trimmed = re.sub(r"[\s_](\d+|ii|iii|iv)$", "", stripped, flags=re.IGNORECASE)
    key = normalise(trimmed.replace("_", " "))

    if key in index:
        return index[key]

    return None


def year_to_bin(year, start):
    lower = start + ((year - start) // BIN_WIDTH) * BIN_WIDTH
    upper = lower + BIN_WIDTH - 1
    return f"{lower}-{upper}"


def main():
    print("loading anime database ...")
    index = build_year_index()
    print(f"  {len(index):,} title/synonym keys with a year")

    print("loading subtitles ...")
    frame = pd.read_csv(SUBS_CSV, usecols=["jp_text", "file_name"],
                        low_memory=False)
    frame = frame.dropna(subset=["jp_text"])
    frame["show"] = frame["file_name"].astype(str).str.replace(
        r"-\d+\.csv$", "", regex=True)
    print(f"  {len(frame):,} lines, {frame['show'].nunique()} shows")

    shows = sorted(frame["show"].unique())
    years = {}
    unmatched = []

    for show in shows:
        year = lookup_year(show, index)

        if year is None:
            unmatched.append(show)
        else:
            years[show] = year

    print(f"  matched {len(years)}/{len(shows)} shows "
          f"({100.0 * len(years) / len(shows):.1f}%)")

    if not years:
        raise SystemExit("no shows matched - check the database file")

    earliest = min(years.values())
    start = earliest - (earliest % BIN_WIDTH)

    print("segmenting Japanese text ...")
    tagger = fugashi.Tagger()

    by_show_tokens = defaultdict(list)
    by_show_episodes = defaultdict(set)

    for show, text, file_name in zip(frame["show"], frame["jp_text"],
                                     frame["file_name"]):
        if show not in years:
            continue

        by_show_episodes[show].add(file_name)

        for word in tagger(str(text)):
            if word.feature.pos1 in DROP_POS:
                continue

            surface = word.surface.strip()

            if surface:
                by_show_tokens[show].append(surface)

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = []

    for show, tokens in by_show_tokens.items():
        if not tokens:
            continue

        year = years[show]
        group = year_to_bin(year, start)

        directory = os.path.join(OUT_DIR, group)
        os.makedirs(directory, exist_ok=True)

        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", show)
        path = os.path.join(directory, safe + ".txt")

        with open(path, "w", encoding="utf-8") as handle:
            handle.write(" ".join(tokens))

        manifest.append({
            "show": show,
            "year": year,
            "bin": group,
            "episodes": len(by_show_episodes[show]),
            "tokens": len(tokens),
            "types": len(set(tokens)),
            "path": path,
        })

    manifest.sort(key=lambda row: (row["year"], row["show"]))

    with open(os.path.join(OUT_DIR, "manifest.csv"),
              "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    with open(os.path.join(OUT_DIR, "unmatched.txt"),
              "w", encoding="utf-8") as handle:
        handle.write("\n".join(unmatched))

    print()
    print(f"{'bin':<12}{'shows':>7}{'episodes':>10}{'tokens':>12}{'types':>10}")
    print("-" * 51)

    grouped = defaultdict(lambda: {"shows": 0, "episodes": 0, "tokens": 0,
                                   "types": set()})

    for row in manifest:
        bucket = grouped[row["bin"]]
        bucket["shows"] += 1
        bucket["episodes"] += row["episodes"]
        bucket["tokens"] += row["tokens"]

    for group in sorted(grouped):
        bucket = grouped[group]
        print(f"{group:<12}{bucket['shows']:>7}{bucket['episodes']:>10}"
              f"{bucket['tokens']:>12,}")

    total_tokens = sum(row["tokens"] for row in manifest)
    print("-" * 51)
    print(f"{'TOTAL':<12}{len(manifest):>7}"
          f"{sum(r['episodes'] for r in manifest):>10}{total_tokens:>12,}")
    print()
    print(f"unmatched shows: {len(unmatched)}")


if __name__ == "__main__":
    main()
