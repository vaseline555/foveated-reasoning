"""Offline paired outcomes and evidence diagnostics; no extra model calls."""

from collections import Counter

from ..answers import parse_answer, parse_final_answer
from ..common import save
from ..readers import reader_content


def answer(text):
    return parse_answer(text)


def diagnose(graph, gold):
    """Explain logged controller behavior without treating structural gain as truth."""
    state = graph.get("graph") or {}
    errors = graph.get("errors", [])
    direct = parse_final_answer(
        reader_content(graph, "direct"),
        mode=graph.get("direct_answer_mode", "native"),
    )
    positive = sum(
        bool(r.get("task_progress", {}).get("newly_supported"))
        or (r.get("realized_report_gain") or 0) > 0
        for r in graph.get("rounds", [])
    )
    decision = graph.get("decision_graph")
    if decision is not None:
        observations = decision["observations"]
        positive = sum(r["realized_gain"] > 0 for r in observations)
        notes = ["Belief gain uses uncalibrated model likelihoods, not verified visual truth."]
        if direct == gold and graph["prediction"] != gold:
            notes.append("Direct was correct; the native final reader returned a wrong answer.")
        if any(not r["observed"] for r in observations):
            notes.append("At least one crop measurement was unobserved and gave no belief gain.")
        if decision["revisions"] > 0:
            notes.append("The graph was revised with additional paid calls.")
        return dict(stop_reason=graph["stop_reason"], positive_gain_rounds=positive,
                    graph_potential=None, resolved_nodes=[r["node_id"] for r in observations if r["observed"]],
                    direct_prediction=direct, error_examples=errors[:3],
                    fact_examples=[r["reported"] for r in observations[:8]], notes=notes,
                    observed_measurements=sum(r["observed"] for r in observations),
                    measurement_attempts=len(observations), belief=decision["belief"],
                    revisions=decision["revisions"])
    goals = graph.get("task_guidance", {}).get("goals", [])
    resolved = (
        [n["id"] for n in goals if n["status"] == "supported_by_reports"]
        if goals
        else state.get("resolved", [])
    )
    notes = []
    stop = graph["stop_reason"]
    if direct == gold and graph["prediction"] != gold:
        notes.append("Direct was correct; the final reader changed it to a wrong answer.")
    if "FORMAT" in stop or "CONTRACT" in stop:
        notes.append("Exploration stopped on an output/contract failure; inspect rejected fields.")
    if any(
        any(tag in e["message"] for tag in ["IDENTITY", "REFERENT", "OBJECT_TYPE", "SUBSTITUTION"])
        for e in errors
    ):
        notes.append(
            "Identity/referent checks rejected at least one report; this does not prove visual absence."
        )
    if "ZERO_GAIN" in stop or "NOVEL" in stop or "STAGNATION" in stop:
        notes.append("Search stalled without new admissible evidence or novel crops.")
    if state.get("potential") == 1 and graph["prediction"] != gold:
        notes.append(
            "All action items were structurally resolved, but the answer was wrong; completion is not truth."
        )
    if positive == 0 and graph["prediction"] == gold:
        notes.append(
            "Correct answer without positive graph gain; graph utility is not demonstrated."
        )
    if positive and graph["prediction"] == gold:
        notes.append(
            "Contract-valid crop evidence accumulated before a correct final answer; no causal ablation was run."
        )
    facts = []
    for node, reports in state.get("reports", {}).items():
        for report in reports.values():
            value = report.get("value")
            facts.append(
                dict(node=node, value=value.get("literal") if isinstance(value, dict) else value)
            )
    return dict(
        stop_reason=stop,
        positive_gain_rounds=positive,
        graph_potential=len(resolved) / len(goals) if goals else state.get("potential"),
        resolved_nodes=resolved,
        direct_prediction=direct,
        error_examples=errors[:3],
        fact_examples=facts[:8],
        notes=notes,
    )


def postmortem(root, cases, results, labels, names):
    groups = dict(
        baseline_correct_graph_wrong=[],
        graph_correct_baseline_wrong=[],
        efficient_graph_success=[],
        near_miss=[],
    )
    diagnostic_counts = Counter()
    for case in cases:
        key = case["key"]
        graph = results[key + "/B"]
        gold = labels[key]
        baselines = {
            arm: results[key + "/" + arm]
            for arm in names
            if arm != "B" and key + "/" + arm in results
        }
        correct = [arm for arm, result in baselines.items() if result["prediction"] == gold]
        wrong = [arm for arm, result in baselines.items() if result["prediction"] != gold]
        diagnosis = diagnose(graph, gold)
        diagnostic_counts.update(diagnosis["notes"])
        row = dict(
            case=key,
            dataset=case["dataset"],
            category=case.get("category", ""),
            question=case["question"],
            image_path=case["image_path"],
            gold=gold,
            predictions={a: r["prediction"] for a, r in dict(baselines, B=graph).items()},
            metrics={
                a: dict(requests=r["calls"], usd=r["logical_usd"])
                for a, r in dict(baselines, B=graph).items()
            },
            diagnosis=diagnosis,
        )
        graph_correct = graph["prediction"] == gold
        if not graph_correct and correct:
            groups["baseline_correct_graph_wrong"].append(dict(row, successful_baselines=correct))
        if graph_correct and wrong:
            groups["graph_correct_baseline_wrong"].append(dict(row, failed_baselines=wrong))
        cheaper = [
            a
            for a in correct
            if graph["calls"] < baselines[a]["calls"]
            and graph["logical_usd"] < baselines[a]["logical_usd"]
        ]
        if graph_correct and cheaper and diagnosis["positive_gain_rounds"]:
            groups["efficient_graph_success"].append(
                dict(row, more_expensive_correct_baselines=cheaper)
            )
        if not graph_correct and diagnosis["direct_prediction"] == gold:
            groups["near_miss"].append(row)
    output = dict(
        groups=groups,
        counts={key: len(rows) for key, rows in groups.items()},
        diagnostic_counts=dict(diagnostic_counts),
        definitions=dict(
            efficient="Correct graph answer, positive evidence gain, fewer requests AND lower cost than at least one correct baseline.",
            near_miss="Direct answer was correct but the graph method's final reader returned a wrong answer.",
        ),
        limitations="Groups overlap. Small balanced samples are not population-weighted benchmark estimates. Evidence consistency is not visual truth or causal benefit.",
    )
    save(root / "postmortem.json", output)
    titles = dict(
        baseline_correct_graph_wrong="Baseline correct, graph wrong",
        graph_correct_baseline_wrong="Graph correct, baseline wrong",
        efficient_graph_success="Efficient graph success",
        near_miss="Near miss",
    )
    lines = ["# Evidence graph postmortem", "", output["limitations"]]
    for key, rows in groups.items():
        lines += ["", f"## {titles[key]} ({len(rows)})", ""]
        if not rows:
            lines.append("No matching case under the stated criterion.")
        for row in rows[:3]:
            predictions = ", ".join(
                f"{names[a]}={value}" for a, value in row["predictions"].items()
            )
            lines.append(f"- {row['dataset']}/{row['case']}: gold={row['gold']}; {predictions}.")
            lines.append("  " + " ".join(row["diagnosis"]["notes"]))
    (root / "postmortem.md").write_text("\n".join(lines) + "\n")
    return output
