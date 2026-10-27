"""Local question/choice vocabulary, not NER claims or image observations."""

import re

STOP = set(
    "a an the what which is are was were be been being of in on at to for from with and or this that it its as by has have do does how i you your answer option letter given choices directly picture image considering".split()
)


def terms(text):
    return list(
        dict.fromkeys(
            token.lower()
            for token in re.findall(r"[A-Za-z0-9]+(?:[-'’][A-Za-z0-9]+)*", text)
            if token.lower() not in STOP and (len(token) > 1 or token.isdigit())
        )
    )


def extract_keywords(question):
    """Run before compiling every plan; alternatives are hypotheses, not facts."""
    lines = question.splitlines()
    choices = {}
    for line in lines[1:]:
        match = re.match(r"\s*(?:\(([A-K])\)|([A-K])[.)])\s*(.+)", line)
        if match:
            choices[match[1] or match[2]] = terms(match[3])
    common = set.intersection(*(set(v) for v in choices.values())) if choices else set()
    return dict(
        method="local_question_choice_tokens",
        question_terms=terms(lines[0] if lines else "")[:32],
        choice_contrasts={
            label: [t for t in ts if t not in common][:10] for label, ts in choices.items()
        },
        shared_choice_terms=sorted(common)[:10],
        source="question_and_choices_only",
        alternatives_are_not_visual_facts=True,
        api_calls=0,
    )
