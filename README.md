# Foveated Reasoning

Crop-based visual reasoning with native FOVEA baselines and an evidence graph,
including a fixed-probability independent query recheck. Python 3.12+ on Linux/macOS.
Inference uses Qwen3-VL-8B-Instruct through OpenRouter with the Parasail BF16 provider;
no local GPU or model-weight download is required.

This repository contains executable source, offline tests, and benchmark download
scripts. Datasets, model responses, experiment histories, credentials, and local
environment files are excluded. It is a clean source export, not a checkpoint of
an already running experiment.

## Install

```bash
git clone https://github.com/vaseline555/foveated-reasoning.git
cd foveated-reasoning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python run_experiment.py self-test
python -m pytest -q
```

The self-test uses simulated responses and makes no paid API calls. Tests that
require private historical experiment artifacts are skipped when those artifacts
are absent; the self-contained tests still run.

## Download datasets

The downloader uses fixed upstream revisions and creates the native layout expected
by the runner. It does not use the inference API or require an OpenRouter key.

```bash
# Download the metadata and images needed for this exact sample.
python scripts/download_datasets.py --samples 10 --seed 42

# Or download all benchmark images. This can use substantial disk/network space.
python scripts/download_datasets.py

# Select individual benchmarks, or fetch metadata only.
python scripts/download_datasets.py --benchmarks vstar mme --samples 5 --seed 42
python scripts/download_datasets.py --benchmarks mme --metadata-only
```

Use the same `--samples`, `--seed`, and optional `--stratified` arguments when
preparing the experiment. Sampling uses question IDs and categories, not labels
or image availability; a missing image is an error, never silently replaced.
Repeated downloads reuse existing files. Changing a sample can add images to the
same directory. TreeBench embeds images in its TSV, so its complete TSV is needed
even for a small sample or `--metadata-only`. MME reads individual ZIP members via
HTTP ranges, with no automatic full-archive fallback.

| Benchmark | Official source | Pinned revision |
| --- | --- | --- |
| V* Bench | [craigwu/vstar_bench](https://huggingface.co/datasets/craigwu/vstar_bench) | `d9ae62c903da0c98336e85c5ee89cd863b04b4da` |
| TreeBench | [HaochenWang/TreeBench](https://huggingface.co/datasets/HaochenWang/TreeBench) | `a4eb437ab71d14f6ab1a4cf05213d0f72f831b71` |
| MME-RealWorld-Lite | [yifanzhang114/MME-RealWorld-Lite](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Lite) | `0e378f64aea87bb2c30dae8bc600f99fae966711` |

Dataset terms and attribution remain with the upstream sources; the code license
does not license those datasets. The downloader records the revision and selected
IDs in `data/<benchmark>/download_manifest.json`.

```text
data/
  vstar/test_questions.jsonl
  vstar/direct_attributes/...
  vstar/relative_position/...
  treebench/TreeBench.tsv
  mme/mme_questions.json
  mme/images/...
```

## Prepare and run the current graph

The current graph controller is selected explicitly with `--graph-controller partition`.
Use `--retest-probability 0` for the control and `0.5` for the fixed recheck condition.
Other controller variants remain available for compatibility; the CLI's historical
default is `python`, which does not perform query rechecks.

```bash
python run_experiment.py prepare \
  --data data --samples 10 --seed 42 \
  --methods graph --graph-controller partition \
  --crop-probe-mode graph-single --retest-probability 0.5 \
  --retry-delay 15 --budget 15 --output results/retest

# These steps do not call the inference API.
python run_experiment.py run results/retest --dry-run
cat results/retest/budget_quote.json

# Supply OPENROUTER_API_KEY through your shell or secret manager.
# .env.example documents the name; .env files are not loaded automatically.
python run_experiment.py run results/retest --approve
```

The last command performs paid inference. Alternatively, use `start` instead of
`run` to start a local background worker, then `status results/retest` to inspect it.
A private file can supply the key using `--api-key-file /path/to/key`.
There are no credentials in this repository.

`--budget` is a **cumulative hard cap** for requests and unresolved failure holds
under the shared account's `results/` directory, not a new allowance for each run.
Final-reader reservations are conservative headroom, not payments. Failed attempts
retain their holds. Every run gets a new output directory with frozen source/input
hashes, and paid execution requires `--approve`. Do not copy an existing paid run
into this clean checkout and assume its accounting or worker state was migrated.

The native metadata retains labels for terminal scoring, but labels and annotated
boxes are not passed into inference. For custom data, use a label-free JSON/JSONL
`--manifest` and provide a separate `--labels` case-key-to-answer JSON file.

## Algorithm

1. Plan question-led visual tasks and their prerequisite DAG. Bind the answer query
   to the question, and preserve prerequisite context in its physical crop.
2. Choose a dependency-ready view using the graph's expected answer-partition gain
   and pending-view cost. Each search step uses one native FOVEA crop/probe, plus
   a separately billed observer of that crop.
3. Admit only structurally supported crop reports. Ordinary facts do not receive
   answer credit. `resolved=true` and a supplied option letter remain model claims,
   not verification of visual truth. Repairs and exploration are bounded.
4. At the first admitted query report, draw once from a reproducible hash of the
   seed and case ID. With fixed probability `p`, reread the **same crop** once,
   without the previous report, answer, native-probe prose, or original image.
5. Retain both fallible reports for the native final reader. Rechecking never
   overwrites the original graph, creates new pixels, or applies majority/latest
   wins. Malformed audit outputs remain paid and do not get model-output retries.
6. Native Direct and final readers arbitrate the available evidence. An actual
   length truncation permits at most one bounded 512-token finalization; ordinary
   invalid answers are not silently retried. All attempts remain accounted for.

The recheck probability is fixed; adaptive scheduling is not implemented. Query
eligibility includes observed reports without a choice mapping. Graph size, depth,
repair count, mapping availability, and crop area are recorded for later analysis.
Matching seeds do not guarantee identical hosted-model responses. No accuracy
improvement or causal benefit is claimed by this source release.

## Native baselines and layout

`--methods greedy mcmc lookahead graph` runs all four methods. With
`--crop-probe-mode graph-single`, the FOVEA baselines keep their native K=3
selection while the graph uses one crop/probe per step. The optional `single`
mode changes all methods and should not be confused with the native baselines.

```text
active_vlm/baselines/    Native FOVEA policies, geometry and prompts
active_vlm/decision/     Partition graph and fixed-probability query recheck
active_vlm/experiment/   Preparation, accounting, execution and reporting
active_vlm/graph/        Earlier graph controllers retained for compatibility
scripts/                Standalone benchmark downloads
tests/                  Offline tests; no inference calls
run_experiment.py       CLI entry point
```

For cloud setup, install the package, download the selected dataset images, and
provide runtime credentials plus HTTPS access to OpenRouter. Dataset downloads
also need Hugging Face and its download CDN. Keep the API provider/model, seed,
sampling policy, and recheck probability fixed during a comparison.

FOVEA baseline code and templates are adapted from
[iamlilAJ/active-vlm](https://github.com/iamlilAJ/active-vlm).
Original attribution is retained in [NOTICE](NOTICE) and [LICENSE](LICENSE).
