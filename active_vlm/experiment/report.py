"""Automatic terminal-only scoring and compact CSV/Markdown output."""

import csv
from pathlib import Path

from ..answers import answer_mode, parse_final_answer, parser_name
from ..common import digest, read, save
from ..protocol import crop_protocol
from ..readers import reader_content
from .accounting import finished, returned, snapshot
from .labels import labels_after_terminal


def metrics(cases, arm, results, labels, scored):
    present = [results[c["key"] + "/" + arm] for c in cases if c["key"] + "/" + arm in results]
    correct = (
        sum(
            r.get("prediction") is not None and r["prediction"] == labels.get(r["case_key"])
            for r in present
        )
        if scored
        else None
    )
    return dict(
        n=len(cases),
        correct=correct,
        accuracy=correct / len(cases) if scored and cases else None,
        attempted_paths=len(present),
        finished_paths=sum(finished(r) for r in present),
        final_reader_completed=sum(returned(r) for r in present),
        missing_paths=len(cases) - len(present),
        failed_paths=sum(finished(r) and r.get("prediction") is None for r in present),
        logical_requests=sum(
            sum(x.get("status", "cached") in {"ok", "cached"} for x in r["receipts"])
            for r in present
        ),
        failed_attempts=sum(r.get("failed_transport_attempts", 0) for r in present),
        logical_known_usd=sum(r["logical_usd"] for r in present),
        logical_unknown_reserve_usd=sum(r.get("logical_unknown_reserve_usd", 0) for r in present),
    )


def report(root):
    end = read(root / "execution_end.json")
    if not end["terminal"]:
        raise RuntimeError("Do not open labels during inference")
    plan = read(root / "plan.json")
    cases = read(root / "inference_manifest.json")
    arms = plan["arms"]
    config = read(root / "config.json")
    protocols = {arm: crop_protocol(config, arm) for arm in arms}
    results = {}
    normalized_saved_answers = {}
    matrix = []
    for case in cases:
        status = {}
        for arm in arms:
            key = case["key"] + "/" + arm
            p = root / "trajectories" / case["key"] / arm / "result.json"
            if not p.exists() and key in plan["reuse"]["completed"]:
                p = Path(plan["reuse"]["completed"][key]["path"])
            r = read(p) if p.exists() else None
            status[arm] = bool(r and returned(r))
            if r:
                final = reader_content(r, "aggregator")
                if final is not None:
                    parsed = parse_final_answer(
                        final,
                        case["allowed_letters"],
                        r.get("final_answer_mode", answer_mode(config)),
                    )
                    if parsed != r["prediction"]:
                        normalized_saved_answers[key] = dict(stored=r["prediction"], parsed=parsed)
                    r["prediction"] = parsed
                else:
                    r["prediction"] = None
                results[key] = r
        matrix.append(dict(case=case["key"], dataset=case["dataset"], returned=status))
    eligible = [c for c, m in zip(cases, matrix, strict=True) if all(m["returned"].values())]
    outcomes = [
        dict(
            case=c["key"],
            arm=a,
            status=results.get(c["key"] + "/" + a, {}).get("status", "not_started"),
            final_readers=bool(
                results.get(c["key"] + "/" + a) and returned(results[c["key"] + "/" + a])
            ),
            stop_reason=results.get(c["key"] + "/" + a, {}).get("stop_reason"),
        )
        for c in cases
        for a in arms
    ]
    save(root / "path_outcomes.json", outcomes)
    save(
        root / "completion.json",
        dict(
            matrix=matrix,
            paired_case_keys=[c["key"] for c in eligible],
            rule="Terminal paired final-reader completion, sealed before reading labels",
        ),
    )
    labels = {}
    source = read(root / "dataset_sources.json")
    physical = [
        read(p) for p in (root / "requests").glob("*.json") if not read(p).get("imported_from")
    ]
    simulation = any(r.get("backend") == "mock" for r in physical)
    if not simulation and source.get("sources") and cases:
        native = [c for c in cases if c["dataset"] in source["sources"]]
        if native:
            labels.update(labels_after_terminal(root, native))
    if not simulation and source.get("labels_path") and cases:
        path = Path(source["labels_path"])
        if digest(path.read_bytes()) != source["labels_sha256"]:
            raise RuntimeError("Label source changed")
        labels.update(read(path))
    scored = bool(cases) and all(c["key"] in labels for c in cases)
    totals = {arm: metrics(cases, arm, results, labels, scored) for arm in arms}
    rows = []
    for case in cases:
        row = dict(case=case["key"], dataset=case["dataset"], gold=labels.get(case["key"]))
        for arm in arms:
            row[arm] = results.get(case["key"] + "/" + arm, {}).get("prediction")
        rows.append(row)
    with (root / "answers.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case", "dataset", "gold", *arms])
        writer.writeheader()
        writer.writerows(rows)
    spent = sum(r.get("cost_usd") or 0 for r in physical)
    held = sum(r["reserve_usd"] for r in physical if r.get("cost_usd") is None)
    summary = dict(
        answer_parser=parser_name(answer_mode(config)),
        final_answer_mode=answer_mode(config),
        normalized_saved_answers=normalized_saved_answers,
        status=end["status"],
        paired_questions=len(eligible),
        target_questions=len(cases),
        totals=totals,
        primary_denominator="all sealed questions; missing, failed and invalid answers remain incorrect",
        path_outcomes=outcomes,
        paired_totals={arm: metrics(eligible, arm, results, labels, scored) for arm in arms},
        crop_probe_protocol=config.get("crop_probe_mode", "native"),
        crop_probe_by_arm=protocols,
        new_physical_requests=len(physical),
        new_successful_physical_requests=sum(r.get("status") == "ok" for r in physical),
        new_failed_physical_attempts=sum(r.get("status") != "ok" for r in physical),
        new_known_usd=spent,
        new_unknown_reserve_usd=held,
        cumulative=snapshot(account_root=read(root / "config.json").get("account_root")),
        simulation=simulation,
        mixed_time_reused_baselines=bool(plan["reuse"]["completed"]),
    )
    summary["by_benchmark"] = {}
    for ds in sorted({c["dataset"] for c in cases}):
        group = [c for c in cases if c["dataset"] == ds]
        summary["by_benchmark"][ds] = {a: metrics(group, a, results, labels, scored) for a in arms}
    graphs = [r for r in results.values() if r["arm"] == "B"]
    if read(root / "config.json").get("engine") == "dependency_graph":
        summary["task_guidance"] = dict(
            intervention="task_dependency_frontier",
            shared_fovea_crop_probe=True,
            reported_support_is_not_verified_visual_truth=True,
            by_arm={
                arm: dict(
                    supported_nodes=sum(
                        n["status"] == "supported_by_reports"
                        for r in results.values()
                        if r["arm"] == arm
                        for n in r.get("task_guidance", {}).get("goals", [])
                    ),
                    unresolved_nodes=sum(
                        n["status"] == "unresolved"
                        for r in results.values()
                        if r["arm"] == arm
                        for n in r.get("task_guidance", {}).get("goals", [])
                    ),
                    repeat_rejections=sum(
                        r.get("no_novel_crop_rejections", 0)
                        for r in results.values()
                        if r["arm"] == arm
                    ),
                )
                for arm in arms
            },
        )
    from collections import Counter

    summary["graph_diagnostics"] = dict(
        stop_reasons=dict(Counter(r["stop_reason"] for r in graphs)),
        crop_rounds=sum(r["crop_rounds"] for r in graphs),
        positive_gain_rounds=sum(
            bool(rd.get("task_progress", {}).get("newly_supported"))
            for r in graphs
            for rd in r["rounds"]
        ),
        supported_nodes=sum(
            n["status"] == "supported_by_reports"
            for r in graphs
            for n in r.get("task_guidance", {}).get("goals", [])
        ),
        repeat_rejections=sum(r.get("no_novel_crop_rejections", 0) for r in graphs),
        reported_support_is_not_verified_visual_truth=True,
        support_basis=(
            "single_report"
            if protocols.get("B", {}).get("mode") == "single"
            else "independent_report_agreement"
        ),
    )
    import re

    summary["output_diagnostics"] = dict(
        invalid_final_answers={
            a: sum(
                returned(results[c["key"] + "/" + a])
                and (
                    results[c["key"] + "/" + a]["prediction"] is None
                    or results[c["key"] + "/" + a]["prediction"] not in c["allowed_letters"]
                )
                for c in cases
                if c["key"] + "/" + a in results
            )
            for a in arms
        },
        probe_tag_omissions=sum(
            r.get("status") == "ok"
            and r.get("stage", "").startswith("probe_")
            and not re.search(
                r"<answer>\s*(yes|no)\s*</answer>", r["message"].get("content") or "", re.I
            )
            for r in physical
        ),
        truncated_responses=sum(r.get("finish_reason") == "length" for r in physical),
        native_reader_recovery_requests=sum(
            r.get("stage", "") in {"direct_length_retry", "aggregator_length_retry"}
            for r in physical
        ),
        native_reader_recoveries=sum(
            d.get("recovery_attempted", False) and d.get("complete_valid_answer", False)
            for r in results.values()
            for d in r.get("reader_diagnostics", [])
        ),
        incomplete_native_reader_outputs=sum(
            not d.get("complete_valid_answer", False)
            for r in results.values()
            for d in r.get("reader_diagnostics", [])
        ),
    )
    if config.get("graph_controller") == "partition":
        observations = [o for r in graphs for o in r["partition_graph"]["observations"]]
        nodes = [n for r in graphs for n in r["task_guidance"]["nodes"]]
        summary["partition_graph_diagnostics"] = dict(
            reported_observations=len(observations),
            admitted_facts=sum(o["observed"] for o in observations),
            supported_fact_nodes=sum(
                n["status"] in {"observed", "observed_unmapped"} for n in nodes
            ),
            mapped_query_nodes=sum(
                n["kind"] == "measure" and n["status"] == "observed" for n in nodes
            ),
            unmapped_query_nodes=sum(n["status"] == "observed_unmapped" for n in nodes),
            unresolved_nodes=sum(n["status"] in {"pending", "unobserved"} for n in nodes),
            complete_paths=sum(r["task_guidance"]["complete"] for r in graphs),
            positive_gain_observations=sum(o["realized_gain"] > 0 for o in observations),
            region_revisions=sum(len(r["partition_graph"]["region_revisions"]) for r in graphs),
            rejected_outputs=sum(len(r["task_guidance"]["plan_rejections"]) for r in graphs),
            structural_support_is_not_visual_truth=True,
        )
        summary["graph_diagnostics"].update(
            support_basis="conditional_crop_reports_not_verified_visual_truth",
            supported_nodes=summary["partition_graph_diagnostics"]["supported_fact_nodes"],
            positive_gain_rounds=sum(
                len(
                    {
                        o["crop_id"]
                        for o in r["partition_graph"]["observations"]
                        if o["realized_gain"] > 0
                    }
                )
                for r in graphs
            ),
        )
    if config.get("graph_controller") == "decision":
        observations = [o for r in graphs for o in r["decision_graph"]["observations"]]
        summary["decision_graph_diagnostics"] = dict(
            measurement_attempts=len(observations),
            observed_measurements=sum(o["observed"] for o in observations),
            positive_belief_gain_measurements=sum(o["realized_gain"] > 0 for o in observations),
            negative_belief_gain_measurements=sum(o["realized_gain"] < 0 for o in observations),
            graph_revisions=sum(max(0, r["decision_graph"]["revisions"]) for r in graphs),
            rejected_outputs=sum(len(r["task_guidance"]["plan_rejections"]) for r in graphs),
            belief_gain_is_not_verified_accuracy=True,
        )
        summary["graph_diagnostics"]["support_basis"] = "not_applicable_binary_model_measurements"
        summary["graph_diagnostics"]["positive_gain_rounds"] = sum(
            o["realized_gain"] > 0 for o in observations
        )
    save(root / "summary.json", summary)
    lines = [
        f"# Results: {'SIMULATION — not measured performance' if simulation else end['status']}",
        f"\nPaired questions: {len(eligible)}/{len(cases)}.\n",
        "Primary denominator: all sealed questions, including failures and missing outputs.\n",
        f"Crop/probe protocol: {summary['crop_probe_protocol']}. "
        + (
            "MCMC/Lookahead multi-candidate selection is disabled; their single-mode crop policy is equivalent to Greedy, not native FOVEA.\n"
            if summary["crop_probe_protocol"] == "single"
            else (
                "FOVEA A/M/L retain native selectors and K3; graph/flat use one crop and one probe per turn.\n"
                if summary["crop_probe_protocol"] == "graph-single"
                else "Native FOVEA selectors and K3.\n"
            )
        ),
        "| Method | Correct | Logical requests | Logical USD |",
        "|---|---:|---:|---:|",
    ]
    for arm, t in totals.items():
        lines.append(
            f"| {plan['names'][arm]} | "
            f"{t['correct'] if scored else 'unscored'}/{t['n']} | "
            f"{t['logical_requests']} | ${t['logical_known_usd']:.6f} |"
        )
    lines += [
        f"\nNew physical requests: {len(physical)}; known ${spent:.6f}; held ${held:.6f}.",
        "\nLogical totals include shared Direct per method; physical cost counts each receipt once.",
        "\nInvalid answers remain failures; incomplete paths are not converted into answers.",
    ]
    for ds, methods in summary["by_benchmark"].items():
        lines += [f"\n## {ds}\n", "| Method | Correct | Requests | USD |", "|---|---:|---:|---:|"]
        for arm, values in methods.items():
            lines.append(
                f"| {plan['names'][arm]} | {values['correct']}/{values['n']} | "
                f"{values['logical_requests']} | ${values['logical_known_usd']:.6f} |"
            )
    (root / "report.md").write_text("\n".join(lines) + "\n")
    if scored and "B" in arms:
        from .postmortem import postmortem

        available = [c for c in cases if all(c["key"] + "/" + a in results for a in arms)]
        postmortem(
            root,
            available if config.get("graph_controller") == "decision" else eligible,
            results,
            labels,
            plan["names"],
        )
    return summary
