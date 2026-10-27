"""Separate the question spine from option hypotheses without semantic dictionaries.

Only explicit, complete label sequences are parsed. If a block is ambiguous, keep
the complete original question rather than invent or truncate candidate text.
This is prompt organization, not entity recognition or visual evidence.
"""

from ..questions import HEADING, INLINE


def question_context(question, letters):
    heading = HEADING.search(question)
    tail = question[heading.end() :] if heading else question
    matches = list(INLINE.finditer(tail))
    labels = [(m[1] or m[2]).upper() for m in matches]
    options = []
    core = question
    if len(labels) >= 2 and labels == list(letters):
        texts = [
            tail[m.end() : matches[i + 1].start() if i + 1 < len(matches) else len(tail)].strip()
            for i, m in enumerate(matches)
        ]
        if all(texts):
            options = [dict(letter=a, text=t) for a, t in zip(letters, texts, strict=True)]
            core = (question[: heading.start()] if heading else tail[: matches[0].start()]).strip()
    return dict(
        question_spine=core,
        option_hypotheses=options,
        option_source="supplied_question_text",
        options_are_visual_evidence=False,
        complete_explicit_option_block=bool(options),
        additional_api_calls=0,
    )
