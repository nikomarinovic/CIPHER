import re
import unicodedata

LATIN = "latin"
CYRILLIC = "cyrillic"
GREEK = "greek"
OTHER = "other"

CZECH_MARKERS = set("ěřůďťň")
CROATIAN_MARKERS = set("čćđšž")


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-ZčćžšđČĆŽŠĐ0-9]+", text.lower())
    return [word for word in words if len(word) > 1]


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def dominant_script(text: str) -> str:
    counts = {LATIN: 0, CYRILLIC: 0, GREEK: 0, OTHER: 0}

    for character in text:
        if not character.isalpha():
            continue

        try:
            name = unicodedata.name(character)
        except ValueError:
            continue

        if "CYRILLIC" in name:
            counts[CYRILLIC] += 1
        elif "GREEK" in name:
            counts[GREEK] += 1
        elif "LATIN" in name:
            counts[LATIN] += 1
        else:
            counts[OTHER] += 1

    if not any(counts.values()):
        return LATIN

    return max(counts, key=counts.get)


def query_language_hint(query: str):
    lowered = query.lower()

    if any(character in CROATIAN_MARKERS for character in lowered):
        return "hr"

    return None


def has_foreign_markers(text: str, target_lang) -> bool:
    lowered = text.lower()
    czech_hits = sum(1 for character in lowered if character in CZECH_MARKERS)

    if target_lang != "cs" and czech_hits >= 3:
        return True

    return False


def phrase_proximity_bonus(query_tokens: list[str], text: str) -> float:
    if len(query_tokens) < 2:
        return 1.0

    text_lower = text.lower()
    pair_hits = 0
    total_pairs = len(query_tokens) - 1

    for index in range(total_pairs):
        first = re.escape(query_tokens[index])
        second = re.escape(query_tokens[index + 1])
        pattern = re.compile(rf"\b{first}\b\W{{1,3}}\b{second}\b")

        if pattern.search(text_lower):
            pair_hits += 1

    if pair_hits == 0:
        return 0.4

    return 1.0 + (pair_hits / total_pairs) * 0.8