"""Read-only structural strata for new Graph artifacts; no model calls."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def summarize(root):
    groups = defaultdict(lambda: dict(cases=0, supported_goals=0, goals=0))
    for path in sorted(Path(root).rglob("composition.json")):
        data = json.loads(path.read_text())
        depth = data["dependency_depth"]
        group = groups[str(depth) if depth < 3 else "3+"]
        group["cases"] += 1
        group["goals"] += len(data["goals"])
        group["supported_goals"] += sum(data["nodes"][key]["status"] == "supported"
                                        for key in data["goals"])
    return dict(strata=dict(groups), note="Model support counts are not partial-answer accuracy. Join independent labels and completed rows for accuracy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(summarize(parser.parse_args().root), indent=2))
