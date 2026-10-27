"""FOVEA selector adapters with isolated randomness and auditable requests."""

import ast
import asyncio
import random
from types import SimpleNamespace

from ..common import digest, seed_for
from ..imaging import validate_seed
from . import geometry, prompts
from .lookahead import make_policy as make_lookahead
from .mcmc import make_policy as make_mcmc


def select_fovea(run, proposal, call_index):
    if not run.crop_protocol["multi_candidate_selectors_enabled"]:
        raise ValueError("MULTI_CANDIDATE_SELECTOR_DISABLED_IN_SINGLE_MODE")
    parent = proposal["anchor"]["parent"]
    box = validate_seed(proposal["anchor"]["box"])
    source = run.images.images[parent]
    method = run.arm
    prefix = f"{run.turn}_{call_index}"
    if method == "M":
        stages = [f"probe_{prefix}_{b}_{k}" for b in range(7) for k in range(3)]
    else:
        stages = []
        for root in range(3):
            stages.append(f"expansion_{prefix}_{root}")
            stages += [f"probe_{prefix}_{root}_{leaf}_{k}" for leaf in range(3) for k in range(3)]
    # Reserve entire selector worst case BEFORE its first dispatch. Early stop
    # releases undispatched slots and quota; paid attempts always remain charged.
    tokens = run.group(stages)
    consumed = set()
    token_by_stage = dict(zip(stages, tokens, strict=True))
    batch = 0
    leaf_batches = {}
    raw, expansions, events = [], [], []

    class Logger:
        def info(self, event, **kw):
            events.append(dict(event=event, **kw))

        warning = info

        def exception(self, *args, **kwargs):
            raise  # do not turn failed paid requests into a zero-score fallback

    class AI:
        def __init__(self, message):
            import json

            self.content = message.get("content")
            self.tool_calls = [
                dict(args=json.loads(c["function"]["arguments"]))
                for c in message.get("tool_calls", [])
            ]

    class Adapter:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            stage = f"expansion_{prefix}_{len(expansions)}"
            message, rec = run.ask(
                stage, messages, token_by_stage[stage], tools=[prompts.INNER_CROP_TOOL]
            )
            consumed.add(stage)
            ai = AI(message)
            alias = ai.tool_calls[0]["args"].get("image", "") if ai.tool_calls else ""
            expansions.append(
                dict(
                    stage=stage,
                    message=message,
                    receipt_id=rec["receipt_id"],
                    original_parent=parent,
                    reported_image_alias=alias,
                    official_image_1_decode_mismatch=parent == "original"
                    and "image_1" in alias.lower(),
                    undefined_image_2_alias=parent == "original" and "image_2" in alias.lower(),
                )
            )
            return ai

        async def abatch(self, inputs):
            nonlocal batch
            results = []
            records = []
            assert len(inputs) == 3
            root = len(expansions) - 1
            leaf = leaf_batches.get(root, 0)
            for k, messages in enumerate(inputs):
                stage = (
                    f"probe_{prefix}_{batch}_{k}"
                    if method == "M"
                    else f"probe_{prefix}_{root}_{leaf}_{k}"
                )
                message, rec = run.ask(stage, messages, token_by_stage[stage])
                consumed.add(stage)
                results.append(SimpleNamespace(content=message.get("content")))
                records.append(dict(message=message, receipt_id=rec["receipt_id"]))
            raw.append(dict(batch=batch, inputs_hash=digest(inputs[0]), records=records))
            batch += 1
            leaf_batches[root] = leaf + 1
            return results

    namespace = dict(
        random=random.Random(
            seed_for(
                run.case["key"],
                f"{method}/local_selector/{prefix}",
                seed=run.transport.config.get("seed", 0),
            )
        ),
        logger=Logger(),
        AIMessage=AI,
        SystemMessage=lambda content: dict(role="system", content=content),
        HumanMessage=lambda content: dict(role="user", content=content),
        get_base64_url_for_qwen_vl=lambda image: run.images.descriptor(image, True),
    )
    for event in [
        "BATCH_REQUEST_COMPLETE",
        "BATCH_REQUEST_ERROR",
        "INVALID_CONFIDENCE_ASSESS",
        "NODE_EVALUATED",
        "BINDED_LLM_CALL_FAILED",
        "BINDED_LLM_NO_TOOL_CALL",
        "INVALID_EXPANSION",
        "INVALID_IMAGE_REF",
        "LOOK_AHEAD_COMPLETE",
    ]:
        namespace[event] = event
    try:
        if method == "M":
            policy = make_mcmc(namespace)(n_iterations=6, n_samples=3)
            selected_box = asyncio.run(
                policy.get_action(
                    Adapter(),
                    run.case["question"],
                    run.images.images["original"],
                    None if parent == "original" else source,
                    box,
                    prompts.template("boed_resolvability.md"),
                )
            )
            evals = [e for e in events if e["event"] == "NODE_EVALUATED"]
            candidates = [run.images.candidate(parent, ast.literal_eval(e["bbox"])) for e in evals]
            scores = [e["score"] for e in evals]
        else:
            policy = make_lookahead(namespace)([1.5, 1.0, 0.8], 3)
            selected_box = asyncio.run(
                policy.get_action(
                    Adapter(),
                    run.case["question"],
                    run.images.images["original"],
                    None if parent == "original" else source,
                    box,
                    prompts.template("look_ahead_crop_inst.md"),
                    prompts.template("look_ahead_resolvability.md"),
                )
            )
            candidates = [
                run.images.candidate(parent, b)
                for b in geometry.generate_candidates(box, [1.5, 1.0, 0.8])
            ]
            scores = ast.literal_eval(
                next(e["values"] for e in events if e["event"] == "LOOK_AHEAD_COMPLETE")
            )
        selected = run.images.candidate(parent, selected_box)
        run.images.commit(selected)
        run.selected.append(selected["id"])
        run.rounds.append(
            dict(
                turn=run.turn,
                call_index=call_index,
                method=method,
                candidates=[run.images.audit(c) for c in candidates],
                scores=scores,
                selected_id=selected["id"],
                selected_index=next(
                    i for i, c in enumerate(candidates) if c["id"] == selected["id"]
                ),
                mean_selected_score=max(scores),
                realized_report_gain=None,
                paid_probe_count=sum(s.startswith("probe_") for s in consumed),
                expansion_calls=len(expansions),
                raw_reports=raw,
                expansions=expansions,
                policy_events=events,
                ledger_after=None,
            )
        )
        return selected, dict(
            status="success",
            output_image=selected["id"],
            selected_coordinates=list(selected_box),
            original_xyxy=selected["original_xyxy"],
        )
    finally:
        run.transport.budget.release_unused(
            [token_by_stage[s] for s in stages if s not in consumed]
        )
