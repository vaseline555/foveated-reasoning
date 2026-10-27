"""Normalize an image-read question and its separately supplied choices."""

import re

LETTERS = "ABCDEFGHIJK"
OPTION = re.compile(r"^\s*(?:option\s+)?(?:\(([A-K])\)|([A-K])[.):])\s*(.*)$", re.I)
INLINE = re.compile(r"(?<!\S)(?:\(([A-K])\)|([A-K])[.):])\s+", re.I)
HEADING = re.compile(r"(?:^|\n|(?<=[?!]))\s*(?:options|choices)\s*:\s*", re.I)


def normalize_options(options):
    if not isinstance(options, list) or not 2 <= len(options) <= len(LETTERS):
        raise ValueError("INVALID_OPTIONS: supply 2..11 choices")
    choices = []
    labelled = []
    for i, text in enumerate(options):
        if not isinstance(text, str) or not 1 <= len(text) <= 200:
            raise ValueError("INVALID_OPTION_TEXT")
        match = OPTION.match(text)
        labelled.append(bool(match))
        if match:
            if (match[1] or match[2]).upper() != LETTERS[i]:
                raise ValueError(
                    "INVALID_OPTION_LABELS: labels must be unique and contiguous from A"
                )
            text = match[3]
        text = text.strip()
        if not text:
            raise ValueError("EMPTY_OPTION_TEXT")
        choices.append(text)
    if any(labelled) and not all(labelled):
        raise ValueError("MIXED_OPTION_LABELS: label every choice or supply plain choice text")
    return choices


def normalize_reading(text, options):
    """Remove a heading/duplicated choice block without inventing OCR content."""
    if not isinstance(text, str) or not 5 <= len(text) <= 600:
        raise ValueError("INVALID_QUESTION_READING")
    choices = normalize_options(options)
    heading = HEADING.search(text)
    body = text[: heading.start()].strip() if heading else text.strip()
    tail = text[heading.end() :].strip() if heading else text
    matches = list(INLINE.finditer(tail))
    if len(matches) >= 2:
        embedded = [
            f"{m[1] or m[2]}. "
            + tail[m.end() : matches[i + 1].start() if i + 1 < len(matches) else len(tail)].strip()
            for i, m in enumerate(matches)
        ]

        def normalized(rows):
            return [re.sub(r"\s+", " ", x).casefold() for x in rows]

        if normalized(normalize_options(embedded)) != normalized(choices):
            raise ValueError("CONFLICTING_OPTION_BLOCKS: text and options disagree")
        if not heading:
            body = tail[: matches[0].start()].strip()
    elif heading and tail:
        raise ValueError(
            "INVALID_EMBEDDED_OPTIONS: keep the question in text and all choices in options"
        )
    elif len(matches) == 1 and re.search(r"(?:^|\n)\s*$", tail[: matches[0].start()]):
        raise ValueError("INCOMPLETE_EMBEDDED_OPTIONS")
    body = re.sub(r"\s+(?:options|choices)\s*:\s*$", "", body, flags=re.I).strip()
    body = re.sub(r"\s+", " ", body)
    if len(body) < 5:
        raise ValueError("MISSING_QUESTION_TEXT")
    return body + "\n" + "\n".join(f"({LETTERS[i]}) {choice}" for i, choice in enumerate(choices))
