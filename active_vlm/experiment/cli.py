"""Prepare a quote, run an approved comparison, and monitor its worker."""

import argparse
import fcntl
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

from ..answers import parser_name
from ..common import MODEL, PROVIDER, REPO, read, save
from ..protocol import crop_protocol
from . import accounting, data, process

METHODS = dict(greedy="A", mcmc="M", lookahead="L", graph="B", flat="F")
NAMES = dict(
    A="FOVEA-Greedy",
    M="FOVEA-MCMC",
    L="FOVEA-Lookahead",
    B="EvidenceGraph",
    F="FOVEA-Flat",
    G="FOVEA-Graph",
)


def prepare(args):
    account = Path(args.account_root).resolve() if args.account_root else REPO
    output = Path(args.output)
    root = (
        (account / output) if args.account_root and not output.is_absolute() else output
    ).resolve()
    if not root.is_relative_to(account / "results"):
        raise ValueError("Use an output directory under results/ for cumulative accounting")
    if root.exists():
        raise FileExistsError("Choose a new output directory")
    if args.manifest:
        path = Path(args.manifest).resolve()
        rows = (
            read(path)
            if path.suffix != ".jsonl"
            else [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        )
        for row in rows:
            if {"label", "answer", "prediction", "Ground truth", "gt_boxes"} & row.keys():
                raise ValueError("Input manifest must be label-free; pass --labels separately")
            image = Path(row["image_path"])
            row["image_path"] = str(image if image.is_absolute() else path.parent / image)
        sources = {}
    else:
        rows, sources = data.catalog(Path(args.data).resolve(), args.benchmarks)
    chosen = data.select(
        rows, args.benchmarks, args.samples, args.seed, args.full, stratified=args.stratified
    )
    root.mkdir(parents=True)
    cases = data.materialize(chosen, root)
    recovery = args.final_answer_mode == "native" and args.native_reader_recovery == "once"
    upper = 262144 * args.input_price / 1e6 + 2048 * args.output_price / 1e6
    provider = dict(
        PROVIDER,
        max_price=dict(prompt=args.input_price, completion=args.output_price, request=0, image=0),
    )
    config = dict(
        model=args.model,
        provider=provider,
        seed=args.seed,
        K=1 if args.crop_probe_mode == "single" else 3,
        crop_probe_mode=args.crop_probe_mode,
        temperature=0.7,
        max_tokens=2048,
        crop_only=True,
        graph_controller=args.graph_controller,
        final_answer_mode=args.final_answer_mode,
        direct_answer_mode=args.final_answer_mode,
        answer_parser=parser_name(args.final_answer_mode),
        request_upper_usd=upper,
        per_case_usd_cap=args.per_question_budget,
        cumulative_cap_usd=args.budget,
        input_price_per_million=args.input_price,
        output_price_per_million=args.output_price,
        min_dispatch_interval_seconds=args.interval,
        pacing=dict(
            mode=args.pacing,
            minimum_seconds=0.5,
            maximum_seconds=120,
            success_window=3,
            decrease_factor=0.8,
            maximum_backoff_seconds=300,
            jitter_fraction=0.1,
        ),
        retry_cooldown_seconds=args.retry_delay,
    )
    if args.graph_controller in {"decision", "partition"}:
        config.update(strict_harness=True, decision_graph=dict(
            cost_weight=1.0, expected_pair_usd=0.003,
            max_measurements=6, max_revisions=2,
        ))
    if args.graph_controller == "partition":
        config["retest_probability"] = args.retest_probability
    if args.deadline_unix is not None:
        config["deadline_unix"] = args.deadline_unix
    if args.account_root:
        config["account_root"] = str(account)
    if recovery:
        config["native_reader_recovery"] = dict(
            max_length_retries=1, max_tokens=512, strategy="finalize", analysis_max_chars=4096
        )
    config.update(
        engine="dependency_graph",
        graph_search_limits=dict(max_probes=6, max_zero_gain=2),
        seed_arm_aliases={"B": "A", "F": "A"},
        task_policy=dict(
            quorum=2 if args.crop_probe_mode == "native" else 1,
            selection="frontier_support_then_native_relevance",
            stop_nudges=1,
            max_identical_invalid_plans=3,
        ),
        failure_policy="continue_terminal_local_failures",
    )
    arms = [METHODS[name] for name in args.methods]
    protocols = {arm: crop_protocol(config, arm) for arm in arms}
    for case in cases:
        case["arm_order"] = sorted(
            arms, key=lambda arm: accounting.digest([args.seed, case["key"], arm])
        )
    mappings = read(Path(args.relocations)) if args.relocations else {}
    reused = accounting.reuse(
        cases, arms, [Path(p).resolve() for p in args.reuse], root, config, mappings
    )
    history = accounting.snapshot(exclude=root, account_root=account)
    pending = reused["pending"]
    if args.graph_first:
        pending.sort(key=lambda path: path.rsplit("/", 1)[1] not in {"B", "F"})
    pending_cases = {path.rsplit("/", 1)[0] for path in pending}
    final_slots = len(pending) + len(pending_cases)
    final_reserve = final_slots * upper
    remaining = args.budget - history["known_or_reserved_usd"]
    atomic_probe_reserves = {
        arm: upper
        * (
            1
            if protocols[arm]["mode"] == "single"
            else {"A": 9, "M": 21, "L": 30, "B": 9, "F": 9}[arm]
        )
        for arm in arms
    }
    first_arm = pending[0].rsplit("/", 1)[1] if pending else None
    # Planning assumptions, not measured price/latency guarantees. Native mode
    # keeps the original selector/K3 costs; single mode has one probe per turn.
    scenarios = []
    for rounds in [1, 3]:
        calls = len(pending_cases)
        for path in pending:
            arm = path.rsplit("/", 1)[1]
            if arm in {"B", "F"} and args.graph_controller in {"python", "decision", "partition"}:
                # One initial region plan, one optional semantic plan, and the
                # common aggregator. Subsequent decisions do not use the API.
                calls += (2 + 2*rounds if args.graph_controller in {"decision", "partition"}
                          else 3 + rounds * (1 if protocols[arm]["mode"] == "single" else 9))
                continue
            round_calls = (
                2
                if protocols[arm]["mode"] == "single"
                else {"A": 10, "M": 22, "L": 31, "B": 10, "F": 10}[arm]
            )
            calls += 2 + rounds * round_calls + (arm in {"B", "F"})
        expected_retests = args.retest_probability * sum(
            path.endswith("/B") for path in pending
        )
        typical = (calls + expected_retests) * 0.003
        scenarios.append(
            dict(
                crop_rounds=rounds,
                uncached_requests=calls,
                expected_additional_retest_requests=expected_retests,
                estimated_usd=typical,
                with_50pct_buffer_usd=typical * 1.5,
                minimum_spacing_minutes=calls * args.interval / 60,
            )
        )
    blockers = []
    if final_reserve > remaining:
        blockers.append("Remaining cumulative budget cannot protect all final readers")
    if first_arm and final_reserve + atomic_probe_reserves[first_arm] + upper > remaining:
        blockers.append(
            "Initial final-reader protection leaves insufficient room for the first selector batch; "
            "use --graph-first for graph-single or reduce the prepared scope"
        )
    if not cases:
        blockers.append("No questions selected")
    quote = dict(
        ready_for_run=not blockers,
        blockers=blockers,
        scope=dict(
            questions=len(cases),
            benchmarks=dict(Counter(c["dataset"] for c in cases)),
            methods=[NAMES[a] for a in arms],
            paths=len(cases) * len(arms),
            reused_paths=len(reused["completed"]),
            new_paths=len(pending),
        ),
        cumulative_hard_cap_usd=args.budget,
        historical=history,
        remaining_usd=remaining,
        physical_final_reserve_usd=final_reserve,
        scenarios=scenarios,
        paid_calls_in_preparation=0,
        graph_controller=args.graph_controller,
        retest_probability=config.get("retest_probability", 0.0),
        maximum_additional_retest_requests=(
            sum(path.endswith("/B") for path in pending) if args.retest_probability else 0
        ),
        crop_probe_protocol=args.crop_probe_mode,
        crop_probe_by_arm=protocols,
        final_answer_mode=args.final_answer_mode,
        native_reader_recovery=config.get("native_reader_recovery"),
        maximum_additional_native_reader_requests=2 * len(pending) if recovery else 0,
        native_mcmc_lookahead_selectors_enabled=args.crop_probe_mode != "single",
        execution_order="graph_first" if args.graph_first else "seeded_interleaved",
        initial_exploration_headroom_usd=remaining - final_reserve,
        largest_atomic_probe_reservation_by_arm_usd=atomic_probe_reserves,
        estimation_assumptions="One/three crop rounds, $0.003 per request planning assumption, "
        "plus 50% buffer; actual tokens, cache hits and repairs vary. All failed attempts "
        "retain a full request reservation. Final reserve is conservative before cache reuse.",
    )
    source = dict(sources=sources)
    source["sampling"] = dict(
        seed=args.seed,
        stratified=args.stratified,
        strata=["category", "subcategory"] if args.stratified else [],
        population_counts=dict(Counter(c["dataset"] for c in rows)),
        selected_strata={
            ds: dict(Counter(c.get("category", "") for c in chosen if c["dataset"] == ds))
            for ds in args.benchmarks
        },
        labels_used_for_selection=False,
    )
    if args.labels:
        path = Path(args.labels).resolve()
        source.update(labels_path=str(path), labels_sha256=accounting.digest(path.read_bytes()))
    for name, value in [
        ("config.json", config),
        ("inference_manifest.json", cases),
        ("dataset_sources.json", source),
        ("budget_quote.json", quote),
        ("plan.json", dict(arms=arms, names=NAMES, reuse=reused)),
    ]:
        save(root / name, value)
    accounting.freeze(root)
    process.progress(
        root,
        status="PREPARED",
        completed_paths=len(reused["completed"]),
        total_paths=len(cases) * len(arms),
        paid_api_calls=0,
    )
    print(json.dumps(dict(output=str(root), **quote), indent=2))
    return root


def check_endpoint(root, config):
    """Verify the reservation's context and price ceilings before paid dispatch."""
    import urllib.request

    url = "https://openrouter.ai/api/v1/models/" + config["model"] + "/endpoints"
    with urllib.request.urlopen(url, timeout=30) as response:
        metadata = json.load(response)
    endpoints = metadata["data"]["endpoints"]
    endpoint = next(
        (e for e in endpoints if e["provider_name"] == "Parasail" and e["quantization"] == "bf16"),
        None,
    )
    if (
        not endpoint
        or endpoint["status"] != 0
        or endpoint["context_length"] > 262144
        or float(endpoint["pricing"]["prompt"]) * 1e6 > config["input_price_per_million"]
        or float(endpoint["pricing"]["completion"]) * 1e6 > config["output_price_per_million"]
    ):
        raise RuntimeError("Pinned model/provider/price or context ceiling unavailable")
    if config.get("final_answer_mode", "native") == "choice" and not {
        "response_format",
        "structured_outputs",
    } <= set(endpoint.get("supported_parameters", [])):
        raise RuntimeError("Pinned endpoint does not support the required choice JSON schema")
    save(root / "endpoint_metadata.json", metadata)


def run(root, approve=False, key_file=None, responder=None, dry_run=False, resume=False):
    from ..baselines.runner import FoveaRun
    from ..budget import BudgetStore
    from ..transport import Transport
    from .report import report

    root = Path(root).resolve()
    if dry_run:
        accounting.verify(root)
        from ..imaging import Images

        for case in read(root / "inference_manifest.json"):
            Images(case, root / "image_audit" / case["key"])
        process.progress(root, status="DRY_RUN_COMPLETE", pid=os.getpid(), paid_api_calls=0)
        print("Inputs and source verified; no API calls.", flush=True)
        return "COMPLETE"
    accounting.approval(root, approve)
    # One worker across the shared cumulative account, including different runs.
    config, plan = read(root / "config.json"), read(root / "plan.json")
    account = Path(config.get("account_root", REPO))
    lock = (account / "results/.execution.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if (root / "physical_ledger.json").exists() and not resume:
            raise RuntimeError("Run already started; prepare a new output with --reuse")
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if key_file:
            key = Path(key_file).read_text().strip()
        if responder is None and not key:
            raise RuntimeError("Set OPENROUTER_API_KEY or pass --api-key-file")
        if responder is None:
            check_endpoint(root, config)
        for proof in list(plan["reuse"]["completed"].values()) + list(
            plan["reuse"]["cache_imports"].values()
        ):
            if accounting.digest(Path(proof["path"]).read_bytes()) != proof["sha256"]:
                raise RuntimeError("Reuse source changed")
        cases = {case["key"]: case for case in read(root / "inference_manifest.json")}
        existing = {}
        for path in plan["reuse"]["pending"]:
            file = root / "trajectories" / path / "result.json"
            if file.exists() and accounting.finished(read(file)):
                existing[path] = read(file)
        if resume:
            from .resume import prepare_resume

            prepare_resume(root)
        budget = BudgetStore(
            root / "physical_ledger.json",
            config,
            accounting.snapshot(exclude=root, account_root=account)["known_or_reserved_usd"],
            config["cumulative_cap_usd"],
            plan["reuse"]["pending"],
        )
        if resume:
            budget.prepare_resume(set(existing))
        transport = Transport(root, budget, config, key, responder, resume=resume)
        completed = len(plan["reuse"]["completed"]) + len(existing)
        final_readers = len(plan["reuse"]["completed"]) + sum(
            accounting.returned(r) for r in existing.values()
        )
        failed_paths = sum(r.get("prediction") is None for r in existing.values())
        total = len(cases) * len(plan["arms"])
        process.progress(
            root,
            status="RUNNING",
            pid=os.getpid(),
            completed_paths=completed,
            finished_paths=completed,
            final_reader_completed_paths=final_readers,
            failed_paths=failed_paths,
            total_paths=total,
            successful_physical_requests=0,
            failed_attempts=0,
            started_unix=read(root / "worker_history.json")["original_started_unix"]
            if resume
            else time.time(),
        )
        transport.update_progress()
        by_benchmark, by_method = Counter(), Counter()
        for path, result in existing.items():
            by_benchmark[result["dataset"]] += 1
            by_method[NAMES[path.rsplit("/", 1)[1]]] += 1
        status = "COMPLETE"
        try:
            for path in plan["reuse"]["pending"]:
                if path in existing:
                    continue
                if (root / "stop_requested.json").exists():
                    status = "STOPPED_BY_USER"
                    break
                case, arm = path.rsplit("/", 1)
                process.progress(root, current_path=path)
                print(f"[{completed + 1}/{total}] {NAMES[arm]} {case}", flush=True)
                if arm in {"B", "F"}:
                    if config.get("graph_controller") == "partition":
                        from ..decision.partition_runner import PartitionGraphRun
                        cls = PartitionGraphRun
                    elif config.get("graph_controller") == "decision":
                        from ..decision.runner import DecisionGraphRun
                        cls = DecisionGraphRun
                    else:
                        from ..graph.runner import GraphRun
                        cls = GraphRun
                else:
                    cls = FoveaRun
                try:
                    trajectory = cls(cases[case], arm, transport, root / "trajectories" / case / arm)
                except Exception as error:
                    if config.get("strict_harness"):
                        from ..errors import HarnessError
                        raise HarnessError(f"initialization:{type(error).__name__}:{error}") from error
                    raise
                result = trajectory.execute()
                terminal = accounting.finished(result)
                completed += terminal
                final_readers += accounting.returned(result)
                failed_paths += terminal and result.get("prediction") is None
                if terminal:
                    budget.finish_path(path)
                    by_benchmark[cases[case]["dataset"]] += 1
                    by_method[NAMES[arm]] += 1
                process.progress(
                    root,
                    completed_paths=completed,
                    finished_paths=completed,
                    final_reader_completed_paths=final_readers,
                    failed_paths=failed_paths,
                    last_stop_reason=result["stop_reason"],
                    completed_by_benchmark=dict(by_benchmark),
                    completed_by_method=dict(by_method),
                )
                if read(budget.path)["stopped"] or not terminal:
                    status = "STOPPED_REQUEST_OR_BUDGET_FAILURE"
                    break
            else:
                if failed_paths:
                    status = "COMPLETE_WITH_FAILURES"
        except KeyboardInterrupt:
            status = "STOPPED_BY_USER"
        except Exception as error:
            status = "STOPPED_" + type(error).__name__
            print(f"{status}: {error}", flush=True)
        finally:
            save(
                root / "execution_end.json",
                dict(terminal=True, allow_new_requests=False, status=status),
            )
            process.progress(root, status=status, completed_paths=completed, current_path=None)
            # A code bug invalidates the whole study. Keep receipts, do not
            # inspect labels before its clean restart is frozen.
            if status != "STOPPED_HarnessError":
                report(root)
        print(f"{status}: {root / 'report.md'}", flush=True)
        return status
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Crop-only FOVEA and evidence graph comparison")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Select samples and estimate cost; no API calls")
    prep.add_argument("--data", default="data")
    prep.add_argument("--manifest")
    prep.add_argument("--labels", help="Separate case-key -> answer JSON, read after inference")
    prep.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["vstar", "treebench", "mme"],
        default=["vstar", "treebench", "mme"],
    )
    prep.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS),
        default=["greedy", "mcmc", "lookahead", "graph"],
    )
    prep.add_argument(
        "--account-root", help="Shared repository holding results/ and its cumulative ledger"
    )
    size = prep.add_mutually_exclusive_group()
    size.add_argument("--samples", type=int, default=5, help="Questions per benchmark")
    size.add_argument("--full", action="store_true", help="All questions")
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument(
        "--stratified",
        action="store_true",
        help="Balance task/category quotas, then diversify subcategories",
    )
    prep.add_argument("--model", default=MODEL)
    prep.add_argument(
        "--graph-controller",
        choices=["python", "model", "decision", "partition"],
        default="python",
        help="decision: costed Bayesian experiments; python: reused region plan; model: model-guided search",
    )
    prep.add_argument(
        "--final-answer-mode",
        choices=["choice", "native"],
        default="native",
        help="choice: answer-only constrained JSON; native: original FOVEA tagged reasoning output",
    )
    prep.add_argument(
        "--retest-probability", type=float, default=0.0,
        help="Fixed probability of one blind query reread; requires partition controller and graph",
    )
    prep.add_argument(
        "--native-reader-recovery",
        choices=["once", "off"],
        default="once",
        help="Native readers: one budgeted 512-token native finalization after length truncation; off replays historical behavior",
    )
    prep.add_argument(
        "--crop-probe-mode",
        choices=["graph-single", "single", "native"],
        default="graph-single",
        help="graph-single: native FOVEA, single graph/flat; single: all methods single; native: all native/K3",
    )
    prep.add_argument(
        "--graph-first",
        action="store_true",
        help="Finish graph/flat paths first, then seeded baselines; preserves shared final-reader protection",
    )
    prep.add_argument(
        "--budget", type=float, default=30, help="Cumulative USD cap including history"
    )
    prep.add_argument("--per-question-budget", type=float, default=3)
    prep.add_argument("--interval", type=float, default=0.5)
    prep.add_argument(
        "--pacing",
        choices=["adaptive", "fixed"],
        default="adaptive",
        help="Adaptive: decrease spacing after successes; back off on 429",
    )
    prep.add_argument("--retry-delay", type=float, default=60)
    prep.add_argument("--deadline-unix", type=float, help="No new HTTP dispatch after this UTC epoch")
    prep.add_argument(
        "--input-price", type=float, default=0.25, help="Maximum USD per million tokens"
    )
    prep.add_argument("--output-price", type=float, default=0.75)
    prep.add_argument(
        "--reuse", nargs="*", default=[], help="Prior run directories; reuse FOVEA only"
    )
    prep.add_argument("--relocations", help="Optional JSON mapping for relocated archived files")
    prep.add_argument("--output", default="results/experiment")
    for command in ["run", "start"]:
        action = commands.add_parser(
            command, help="Run approved output" + (" in background" if command == "start" else "")
        )
        action.add_argument("output", type=Path)
        action.add_argument(
            "--approve", action="store_true", help="Authorize the prepared cost/scope"
        )
        action.add_argument("--api-key-file", type=Path, help="Private key file; key never logged")
        action.add_argument(
            "--dry-run", action="store_true", help="Verify prepared files without API calls"
        )
        action.add_argument(
            "--resume",
            action="store_true",
            help="Replay exact cached responses and continue a stopped run",
        )
    for command in ["status", "stop", "report"]:
        commands.add_parser(command).add_argument("output", type=Path)
    check = commands.add_parser(
        "self-test", help="Check all four methods offline, without API credentials"
    )
    check.add_argument(
        "--crop-probe-mode", choices=["graph-single", "single", "native"], default="graph-single"
    )
    args = parser.parse_args(argv)
    if args.command == "self-test":
        from .selftest import self_test

        print(json.dumps(self_test(args.crop_probe_mode), indent=2))
        return
    if args.command == "prepare":
        if not math.isfinite(args.retest_probability) or not 0 <= args.retest_probability <= 1:
            parser.error("retest probability must be finite and between 0 and 1")
        if args.retest_probability and (
            args.graph_controller != "partition" or "graph" not in args.methods
        ):
            parser.error("retesting requires --graph-controller partition and --methods graph")
        for name in ["budget", "per_question_budget", "input_price", "output_price"]:
            if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
                parser.error(name + " must be positive and finite")
        if args.samples <= 0 or any(
            not math.isfinite(x) or x < 0 for x in [args.interval, args.retry_delay]
        ):
            parser.error(
                "samples must be positive; request intervals must be finite and nonnegative"
            )
        return prepare(args)
    root = args.output.resolve()
    if args.command == "run":
        status = run(
            root, args.approve, args.api_key_file, dry_run=args.dry_run, resume=args.resume
        )
        if status != "COMPLETE":
            raise SystemExit(2)
    elif args.command == "start":
        print(
            json.dumps(
                process.start(root, args.approve, args.api_key_file, args.dry_run, args.resume),
                indent=2,
            )
        )
    elif args.command == "status":
        print(json.dumps(process.status(root), indent=2))
    elif args.command == "stop":
        print(process.stop(root))
    elif args.command == "report":
        from .report import report

        print(json.dumps(report(root), indent=2))
