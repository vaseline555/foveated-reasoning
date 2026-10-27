"""Credential-free execution check through the same four-method worker."""

import io
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from ..common import MODEL, canonical, read, save


def self_test(mode="graph-single"):
    from .cli import main, run

    with TemporaryDirectory(prefix="active-vlm-check-") as directory:
        account = Path(directory)
        image = account / "image.png"
        Image.new("RGB", (300, 200), "blue").save(image)
        manifest = account / "manifest.json"
        save(
            manifest,
            [
                dict(
                    dataset="vstar",
                    id="offline-check",
                    image_path=str(image),
                    question="What color is the image?\n(A) Blue\n(B) Red",
                )
            ],
        )

        def respond(request, meta):
            stage = meta["stage"]
            message = dict(role="assistant", content="<answer>A</answer>")
            if stage in {"direct", "aggregator"} and request.get("response_format"):
                message["content"] = '{"answer":"A"}'
            if stage.startswith("probe_"):
                message["content"] = "<answer>Yes</answer>"
            if stage == "planner_1" or stage.startswith("expansion_"):
                inner = stage.startswith("expansion_")
                args = dict(coordinates=[200, 200, 600, 600])
                args["image" if inner else "image_url"] = "image_1" if inner else "original"
                message.update(
                    content="Inspect the region.",
                    tool_calls=[
                        dict(
                            id="offline-crop",
                            type="function",
                            function=dict(
                                name="crop" if inner else "cropping", arguments=canonical(args)
                            ),
                        )
                    ],
                )
            if request.get("tools", [{}])[0].get("function", {}).get("name") == "plan_regions":
                properties = request["tools"][0]["function"]["parameters"]["properties"]
                message = dict(
                    role="assistant",
                    content="Initial regions.",
                    tool_calls=[
                        dict(
                            id="regions",
                            type="function",
                            function=dict(
                                name="plan_regions",
                                arguments=canonical({k: [200, 200, 600, 600] for k in properties}),
                            ),
                        )
                    ],
                )
            return dict(
                id="offline-" + meta["request_hash"],
                model=MODEL,
                provider="Parasail",
                usage=dict(cost=0.001),
                choices=[dict(finish_reason="stop", message=message)],
            )

        with redirect_stdout(io.StringIO()):
            root = main(
                [
                    "prepare",
                    "--manifest",
                    str(manifest),
                    "--benchmarks",
                    "vstar",
                    "--full",
                    "--account-root",
                    str(account),
                    "--output",
                    "results/check",
                    "--pacing",
                    "fixed",
                    "--crop-probe-mode",
                    mode,
                    "--interval",
                    "0",
                ]
            )
            status = run(root, approve=True, responder=respond)
        summary = read(root / "summary.json")
        if status != "COMPLETE" or set(summary["totals"]) != {"A", "M", "L", "B"}:
            raise RuntimeError("Offline execution check did not finish all four methods")
        if not summary["simulation"] or summary["new_failed_physical_attempts"]:
            raise RuntimeError("Offline execution check failed")
        return dict(
            status="PASS",
            simulation=True,
            crop_probe_mode=mode,
            graph_controller="python",
            native_selectors_enabled=mode != "single",
            crop_probe_by_arm=summary["crop_probe_by_arm"],
            network_requests=0,
            methods=["FOVEA-Greedy", "FOVEA-MCMC", "FOVEA-Lookahead", "EvidenceGraph"],
            note="Checks execution only; not measured accuracy or API availability",
        )
