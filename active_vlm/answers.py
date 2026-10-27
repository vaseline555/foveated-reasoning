"""Parse an explicit option label, never infer an answer from its meaning."""

import json
import re

PARSER_NAME = "explicit_tagged_option_label"
CHOICE_PARSER_NAME = "strict_choice_json"


def answer_mode(config):
    mode = config.get("final_answer_mode", "native")
    if mode not in {"native", "choice"}:
        raise ValueError("INVALID_FINAL_ANSWER_MODE")
    return mode


def parser_name(mode):
    return CHOICE_PARSER_NAME if mode == "choice" else PARSER_NAME


def direct_mode(config):
    # Historical choice runs bounded only the final judge. Keep their Direct
    # protocol explicit so reuse cannot silently mix two reader contracts.
    return answer_mode(dict(final_answer_mode=config.get("direct_answer_mode", "native")))


def choice_schema(allowed):
    try:
        letters = list(allowed)
    except TypeError as error:
        raise ValueError("INVALID_ALLOWED_CHOICES") from error
    if (
        not letters
        or any(not isinstance(x, str) or len(x) != 1 or x not in "ABCDEFGHIJK" for x in letters)
        or len(set(letters)) != len(letters)
    ):
        raise ValueError("INVALID_ALLOWED_CHOICES")
    return dict(
        type="object",
        properties=dict(answer=dict(type="string", enum=letters)),
        required=["answer"],
        additionalProperties=False,
    )


def parse_final_answer(text, allowed="ABCDEFGHIJK", mode="native"):
    if mode == "native":
        return parse_answer(text, allowed)
    if mode != "choice":
        raise ValueError("INVALID_FINAL_ANSWER_MODE")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("DUPLICATE_FINAL_ANSWER_KEY")
            result[key] = value
        return result

    try:
        obj = json.loads(text, object_pairs_hook=unique_keys)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or set(obj) != {"answer"}:
        return None
    value = obj["answer"]
    return value if isinstance(value, str) and len(value) == 1 and value in allowed else None


def parse_answer(text, allowed="ABCDEFGHIJK"):
    tags = re.findall(r"<answer>\s*(.*?)\s*</answer>", text or "", re.S | re.I)
    if not tags:
        return None
    value = tags[-1].strip().strip("`*").strip()
    if len(value) == 1 and value.upper() in allowed:
        return value.upper()
    # Accept (A) text, A. text, A: text, and Option A; reject A or B and prose
    # beginning with the article 'a'. Multiple explicit labels remain ambiguous.
    match = re.match(r"^(?:option\s+)?(?:\(([A-K])\)|([A-K])[.):]|([A-K])$)", value, re.I)
    if not match:
        return None
    letter = next(g for g in match.groups() if g).upper()
    rest = value[match.end() :].strip()
    if re.match(r"^(?:(?:or|and)\b|/)", rest, re.I) or re.search(
        r"\([A-K]\)|\b(?:option|choice)\s+[A-K]\b|(?:^|[\s;/])[A-K][.):](?:\s|$)", rest, re.I
    ):
        return None
    return letter if letter in allowed else None
