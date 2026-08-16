"""
Build plain-text corpora of the Bible, Quran and Bhagavad Gita in their
original languages, from the cloned source repositories.

Outputs one .txt file per corpus into scripture_corpus/, plus a manifest.
"""

import json
import os
import re
import unicodedata


SRC = "scripture_src"
OUT = "scripture_corpus"

# Hebrew combining marks: cantillation (te'amim) and vowel points (niqqud).
HEBREW_MARKS = re.compile(r"[\u0591-\u05BD\u05BF-\u05C7]")

# Arabic diacritics (tashkeel), Quranic annotation signs, and tatweel.
ARABIC_MARKS = re.compile(r"[\u064B-\u0652\u0670\u06D6-\u06ED\u08F0-\u08FF\u0640]")

# Devanagari: danda, double danda, digits, avagraha handled separately.
DEVANAGARI_JUNK = re.compile(r"[\u0964\u0965\u0966-\u096F0-9.\u093D]")


def strip_marks(text, pattern):
    normalised = unicodedata.normalize("NFC", text)
    return pattern.sub("", normalised)


def build_hebrew(split_morphemes):
    """
    Westminster Leningrad Codex, consonantal text.

    The source marks prefix morphemes (prepositions, articles, conjunctions)
    with a forward slash. split_morphemes=False keeps each orthographic word
    whole; True splits at those boundaries.
    """
    path = os.path.join(SRC, "openscriptures_morphhb/oxlos-import/wlc.txt")
    words = []

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")

            if len(parts) < 3:
                continue

            surface = parts[2]
            surface = strip_marks(surface, HEBREW_MARKS)
            surface = surface.replace("\u05BE", "/")
            surface = surface.strip("\u05C3 ")

            if split_morphemes:
                pieces = surface.split("/")
            else:
                pieces = [surface.replace("/", "")]

            for piece in pieces:
                cleaned = piece.strip()
                if cleaned:
                    words.append(cleaned)

    return words


def build_greek():
    """SBLGNT, using the normalised word-form column."""
    directory = os.path.join(SRC, "morphgnt_sblgnt")
    words = []

    for name in sorted(os.listdir(directory)):
        if not name.endswith("-morphgnt.txt"):
            continue

        with open(os.path.join(directory, name), encoding="utf-8") as handle:
            for line in handle:
                columns = line.split()

                if len(columns) < 7:
                    continue

                words.append(columns[5])

    return words


def build_arabic():
    """Quran, Arabic text with diacritics stripped."""
    path = os.path.join(SRC, "risan_quran-json/dist/quran.json")

    with open(path, encoding="utf-8") as handle:
        chapters = json.load(handle)

    words = []

    for chapter in chapters:
        for verse in chapter["verses"]:
            text = strip_marks(verse["text"], ARABIC_MARKS)

            for token in text.split():
                cleaned = token.strip("\u06DD\u06DE()[]،.")
                if cleaned:
                    words.append(cleaned)

    return words


def build_sanskrit():
    """Bhagavad Gita, Devanagari verse text."""
    path = os.path.join(SRC, "gita_gita/data/verse.json")

    with open(path, encoding="utf-8") as handle:
        verses = json.load(handle)

    words = []

    for verse in verses:
        text = verse.get("text") or ""
        text = text.replace("\\n", " ")
        text = text.replace("\n", " ")
        text = DEVANAGARI_JUNK.sub(" ", text)

        for token in text.split():
            cleaned = token.strip()
            if cleaned:
                words.append(cleaned)

    return words


def write_corpus(name, words):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".txt")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(" ".join(words))

    unique = len(set(words))
    print(f"{name:<28} tokens={len(words):>8,}  types={unique:>7,}  "
          f"ttr={unique / len(words):.4f}")

    return path


def main():
    hebrew_whole = build_hebrew(split_morphemes=False)
    hebrew_split = build_hebrew(split_morphemes=True)
    greek = build_greek()
    arabic = build_arabic()
    sanskrit = build_sanskrit()

    write_corpus("hebrew_ot_whole", hebrew_whole)
    write_corpus("hebrew_ot_morphemes", hebrew_split)
    write_corpus("greek_nt", greek)
    write_corpus("arabic_quran", arabic)
    write_corpus("sanskrit_gita", sanskrit)

    bible_whole = hebrew_whole + greek
    write_corpus("bible_combined", bible_whole)


if __name__ == "__main__":
    main()
