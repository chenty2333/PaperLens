from __future__ import annotations

import re


TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_+./%-]{2,}|[\u4e00-\u9fff]+", re.IGNORECASE)
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
GROUNDING_STOPWORDS = {
    "about",
    "also",
    "and",
    "are",
    "author",
    "authors",
    "based",
    "been",
    "being",
    "claim",
    "claims",
    "does",
    "evidence",
    "for",
    "from",
    "has",
    "have",
    "into",
    "its",
    "new",
    "not",
    "paper",
    "propose",
    "proposed",
    "proposes",
    "result",
    "results",
    "show",
    "shows",
    "source",
    "study",
    "that",
    "the",
    "their",
    "these",
    "this",
    "those",
    "using",
    "was",
    "were",
    "with",
    "work",
}


def text_overlaps_any_reference(text: str, references: list[str]) -> bool:
    text_tokens = meaningful_tokens(text)
    if not text_tokens:
        return False
    normalized_text = normalize_for_substring(text)
    for reference in references:
        normalized_reference = normalize_for_substring(reference)
        shorter_length = min(len(normalized_text), len(normalized_reference))
        if (
            shorter_length >= 24
            and normalized_reference
            and (normalized_reference in normalized_text or normalized_text in normalized_reference)
        ):
            return True
        reference_tokens = meaningful_tokens(reference)
        overlap = text_tokens & reference_tokens
        if len(overlap) >= 2 or any(is_specific_token(token) for token in overlap):
            return True
    return False


def meaningful_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in TOKEN_PATTERN.findall(text.lower()):
        if CJK_PATTERN.fullmatch(match):
            tokens.update(cjk_ngrams(match))
            continue
        token = match.strip("._-/")
        if token and token not in GROUNDING_STOPWORDS:
            tokens.add(token)
    return tokens


def cjk_ngrams(text: str) -> set[str]:
    if len(text) < 2:
        return set()
    tokens = set()
    for size in (2, 3, 4):
        if len(text) >= size:
            tokens.update(text[index : index + size] for index in range(len(text) - size + 1))
    if len(text) <= 8:
        tokens.add(text)
    return tokens


def is_specific_token(token: str) -> bool:
    return len(token) >= 8 or any(character.isdigit() for character in token)


def normalize_for_substring(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()
