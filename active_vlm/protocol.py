"""Explicit crop/probe protocol; native FOVEA remains separately reproducible."""


def crop_protocol(config, arm=None):
    setting = config.get("crop_probe_mode", "native")
    if setting not in {"single", "native", "graph-single"}:
        raise ValueError("INVALID_CROP_PROBE_MODE")
    if setting == "graph-single" and arm not in {"A", "M", "L", "B", "F", "G"}:
        raise ValueError("METHOD_REQUIRED_FOR_GRAPH_SINGLE_PROTOCOL")
    mode = (
        ("single" if arm in {"B", "F", "G"} else "native") if setting == "graph-single" else setting
    )
    single = mode == "single"
    k, quorum = (1, 1) if single else (3, 2)
    baseline_k = 1 if setting == "single" else 3
    if type(config.get("K", 3)) is not int or config.get("K", 3) != baseline_k:
        raise ValueError(f"PROTOCOL_K_MISMATCH: {setting} requires K={baseline_k}")
    policy = config.get("task_policy", {})
    if not isinstance(policy, dict):
        raise ValueError("INVALID_TASK_POLICY")
    graph_quorum = 2 if setting == "native" else 1
    required = policy.get("quorum", graph_quorum)
    if type(required) is not int or required != graph_quorum:
        raise ValueError(f"PROTOCOL_QUORUM_MISMATCH: {setting} requires quorum={graph_quorum}")
    repair_limit = policy.get("max_identical_invalid_plans", 3)
    if type(repair_limit) is not int or not 1 <= repair_limit <= 10:
        raise ValueError("INVALID_PLAN_REPAIR_LIMIT: expected an integer in1..10")
    return dict(
        mode=mode,
        K=k,
        quorum=quorum,
        candidates_per_crop=1 if single else 3,
        multi_candidate_selectors_enabled=not single,
        support_basis="single_report" if single else "independent_report_agreement",
    )
