"""Native readers with one budgeted finalization of an incomplete draft."""

import re
from collections import Counter
from copy import deepcopy

from .answers import parse_final_answer
from .budget import BudgetStop

RETRY_STAGES = {"direct_length_retry": "direct", "aggregator_length_retry": "aggregator"}


def reader_content(result, stage):
    """Report the effective reader, never rescore an explicitly invalid prefix."""
    diagnostic = next(
        (d for d in result.get("reader_diagnostics", []) if d["stage"] == stage), None
    )
    if diagnostic and not diagnostic.get("complete_valid_answer"):
        return None
    selected = stage + "_length_retry" if diagnostic and diagnostic["recovery_attempted"] else stage
    return next(
        (
            e.get("message", {}).get("content")
            for e in reversed(result.get("events", []))
            if e.get("stage") == selected
        ),
        None,
    )


def recovery_policy(config):
    policy = config.get("native_reader_recovery")
    if policy is None:
        return dict(max_length_retries=0, max_tokens=config.get("max_tokens", 2048))
    if (
        not isinstance(policy, dict)
        or not {"max_length_retries", "max_tokens"} <= set(policy)
        or set(policy) - {"max_length_retries", "max_tokens", "strategy", "analysis_max_chars"}
        or type(policy["max_length_retries"]) is not int
        or policy["max_length_retries"] not in {0, 1}
        or type(policy["max_tokens"]) is not int
        or policy.get("strategy", "same_input") not in {"same_input", "finalize"}
    ):
        raise ValueError("INVALID_NATIVE_READER_RECOVERY_POLICY")
    if policy.get("strategy", "same_input") == "finalize":
        if (
            not 128 <= policy["max_tokens"] <= 2048
            or type(policy.get("analysis_max_chars")) is not int
            or not 256 <= policy["analysis_max_chars"] <= 8192
        ):
            raise ValueError("INVALID_NATIVE_READER_FINALIZATION_POLICY")
    elif (
        not config.get("max_tokens", 2048) <= policy["max_tokens"] <= 8192
        or "analysis_max_chars" in policy
    ):
        raise ValueError("INVALID_NATIVE_READER_RECOVERY_POLICY")
    elif policy["max_length_retries"] and policy["max_tokens"] <= config.get("max_tokens", 2048):
        raise ValueError("NATIVE_READER_RECOVERY_REQUIRES_A_LARGER_TOKEN_CAP")
    return dict(policy)


def repetition_diagnostic(text):
    """Measure repeated long passages without interpreting any visual claim."""
    tokens = re.findall(r"\S+", (text or "").lower())
    # Repeated 64-token windows plus paragraph repetition catch both a prose
    # loop and repeated short-paragraph cycles, without flagging a repeated
    # target name or answer label as a reasoning loop.
    windows = Counter(tuple(tokens[i : i + 64]) for i in range(max(0, len(tokens) - 63)))
    paragraphs = Counter(
        re.sub(r"\s+", " ", p).strip().lower()
        for p in re.split(r"\n\s*\n", text or "")
        if len(p.split()) >= 24
    )
    maximum = max([1, *windows.values(), *paragraphs.values()])
    return dict(loop_detected=maximum >= 3, max_passage_repetitions=maximum)


def finalization_messages(messages, draft, allowed, max_chars):
    """A bounded fallible draft, not a fresh restart or a source of evidence."""
    paragraphs, seen = [], set()
    for paragraph in re.split(r"\n\s*\n", draft or ""):
        key = re.sub(r"\s+", " ", paragraph).strip().lower()
        if key and key not in seen:
            seen.add(key)
            paragraphs.append(paragraph)
    excerpt = "\n\n".join(paragraphs)[:max_chars]
    return [
        *deepcopy(messages),
        dict(role="assistant", content="Incomplete, fallible draft:\n" + excerpt),
        dict(
            role="user",
            content=(
                "Finalize the answer now using the original image, question and unchanged "
                "choice labels above. The draft is incomplete and may be wrong; it is not "
                "additional evidence. Do not restart the analysis, repeat interpretations "
                "or quote the draft. Give at most two short reasoning sentences, then one "
                "complete native <answer>LETTER</answer> tag. Allowed letters: "
                + ", ".join(allowed)
                + ". Preserve distinct labels even when their choice texts are identical."
            ),
        ),
    ]


def read(run, stage, messages, token, mode, schema=None):
    """Keep the first native wire unchanged; never reuse a truncated answer."""
    message, record = run.ask(stage, messages, token, schema=schema)
    policy = recovery_policy(run.transport.config)
    if mode != "native" or not policy["max_length_retries"]:
        return message, None  # Exact historical behavior, including old errors.
    diagnostic = dict(
        stage=stage,
        initial_finish=record["finish_reason"],
        initial_request_hash=record["request_hash"],
        recovery_attempted=False,
        recovery_strategy=policy.get("strategy", "same_input"),
        initial_repetition=repetition_diagnostic(message.get("content")),
    )
    run.reader_diagnostics.append(diagnostic)
    if record["finish_reason"] == "length":
        run.error(stage, ValueError("NATIVE_READER_TRUNCATED"))
        recovery_stage = stage + "_length_retry"
        try:
            recovery_token = run.group([recovery_stage])[0]
        except BudgetStop as error:
            # The ordinary final reader has its protected slot. Optional
            # recovery must not consume that slot or expand any monetary cap.
            diagnostic["recovery_skipped"] = str(error)
        else:
            diagnostic["recovery_attempted"] = True
            recovery_messages = messages
            if policy.get("strategy") == "finalize":
                recovery_messages = finalization_messages(
                    messages,
                    message.get("content"),
                    run.case["allowed_letters"],
                    policy["analysis_max_chars"],
                )
            try:
                message, record = run.ask(recovery_stage, recovery_messages, recovery_token)
            finally:
                run.transport.budget.release_unused([recovery_token])
            diagnostic.update(
                recovery_stage=recovery_stage,
                recovery_finish=record["finish_reason"],
                recovery_request_hash=record["request_hash"],
                recovery_repetition=repetition_diagnostic(message.get("content")),
            )
    valid = (
        record["finish_reason"] == "stop"
        and parse_final_answer(message.get("content"), run.case["allowed_letters"], "native")
        is not None
    )
    diagnostic["complete_valid_answer"] = valid
    if not valid:
        run.error(stage, ValueError("NO_COMPLETE_NATIVE_READER_ANSWER"))
    return message, valid
