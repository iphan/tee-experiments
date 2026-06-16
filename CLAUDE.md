# Claude Code Instructions

## Project

This repository is a **replication and extension of Messing (2026), *Total Eval
Error* (TEE)**. TEE shows that standard LLM-eval confidence intervals are
systematically too narrow because they ignore pipeline variance — prompt
phrasing, judges, temperature, and their interactions — which does not shrink as
you add items. The paper demonstrates this only on **English, frontier models**;
this project reproduces the method and extends it to two regimes the paper flags
as untested: **non-frontier models** and **non-English (French)**.

The repo is built on the
[inspect-eval-template](https://github.com/ArcadiaImpact/inspect-eval-template).
Each evaluation is an [Inspect AI](https://inspect.aisi.org.uk/) task that
generates responses under a **factorial of prompt variants × replications**, so
the variance can be decomposed with the
[`totalevalerror`](https://github.com/SolomonMg/totalevalerror) R package.

### How a TEE eval works (the pattern to follow)

A TEE eval is an Inspect task that:

- reads a **fixed, frozen item set** (the same items are reused across every
  generation cell, so the contrast isn't confounded by item sampling);
- expands each item into one sample per **prompt variant** (`v_0`..`v_{n-1}`;
  `v_0` is the canonical reference variant), giving N items × V variants samples;
- runs each sample for **R replications via Inspect epochs**, at **temperature
  > 0** (replicate noise only exists above 0);
- carries `item_id`, `variant_id`, `language`, and any strata (e.g.
  subject/category) in the sample metadata;
- scores to a **binary correct/incorrect**, recording parse failures distinctly
  (they otherwise masquerade as replicate noise).

The `.eval` log then exports to long format
(`item_id, variant_id, epoch, model, language, correct`) for the R-side
`tee_design → tee_decompose → tee_se_compare → tee_dstudy` pipeline.

#### Exporting results (.eval → long-format CSV/.rda)

`src/utils/tee_export.py` reads one or more `.eval` logs and emits the
long-format table `tee_design()` expects (one row per item × variant ×
replication). It is generic across TEE evals: it reads the standard
sample-metadata keys (`item_id`, `variant_id`, `category`, `language`,
`subject`), the model and temperature from the log, and the replication index
from the Inspect epoch. Scores map to a binary `outcome`; parse failures
(`NOANSWER`) become `NA` so `tee_design()` drops them, and the run prints a
per-cell parse-failure report.

```sh
uv run python -m utils.tee_export --logs logs/ --out exports/tee_long.csv --rda
```

- `--logs` — log directories, globs, or files to read (default: `logs/`); accepts
  multiple entries.
- `--out` — output CSV path (default: `exports/tee_long.csv`).
- `--rda` — also write a `.rda` alongside the CSV by shelling out to
  `scripts/long_csv_to_rda.R` (requires `Rscript` on PATH).
- `--name` — R object name inside the `.rda` (default: the `--out` file stem).

`scripts/long_csv_to_rda.R` is base-R only and can also be run standalone:
`Rscript scripts/long_csv_to_rda.R <csv> <rda_out> [name]`.

### Methodological commitments (from the paper)

- **Prompt variants vary instruction/framing only** — task, output format, and
  reasoning mode are held constant (Assumption 1). They are not question
  rewordings.
- **Anchor conclusions on the item × prompt interaction** (estimated from N×V
  cells → tight CIs). `σ²_prompt` comes from only V levels and is intrinsically
  imprecise; report it with its CI but don't hang conclusions on cross-cell
  differences in it.
- **Design floors:** N ≥ 30 items, **V ≥ 3** variants, R ≥ 5 replications for
  reliable estimation. If call-constrained, cut R before V or N.
- **Report the parse/score rate per cell** — parse failures look like replicate
  noise.
- **Linear probability model (LPM)** for the main decomposition; logit as a
  robustness check near the accuracy floor/ceiling.
- **Don't filter items for variance or difficulty** — representative sampling is
  the whole point.

### Credentials / paid runs

Generation makes **paid API calls** (e.g. via OpenRouter; key in
`OPENROUTER_API_KEY`). The assistant builds and smoke-tests harnesses — always
with `--model mockllm/model` for free smoke tests — but **does not execute paid
runs or handle the key**; the user runs those.

## Running Commands

This project uses [uv](https://docs.astral.sh/uv/) (by Astral) for package
management. **Do not use `pip install`, `python -m venv`,
`source .venv/bin/activate`, or bare `python`/`pytest` commands.** Always use
`uv`:

- `uv sync` — install/sync dependencies (replaces `pip install -e .`)
- `uv run pytest ...` — run tests (not `pytest` or `python -m pytest`)
- `uv run python ...` — run Python scripts (not `python` or `python3`)
- `uv run inspect eval src/<eval>/<file>.py@<task>` — run an evaluation (use the
  file-path form; the `<eval>/<task>` registry form does not resolve locally
  because the distribution is named `inspect-eval-template`, not the eval
  package)
- `uv run ruff ...` — run the linter
- `uv run mypy ...` — run the type checker

Note: `uv` is Astral's Python package manager. It is not related to `uvicorn`
(an ASGI web server) — do not confuse them.

## Contributing

For development setup, submission requirements, and contribution guidelines, see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Coding Style

When writing or modifying code in this repository, follow the guidelines in
[BEST_PRACTICES.md](BEST_PRACTICES.md). Pay particular attention to the
[Writing comments](BEST_PRACTICES.md#writing-comments) section before adding any
comments.

## Evaluation Checklist

When creating or reviewing evaluations, refer to
[EVALUATION_CHECKLIST.md](EVALUATION_CHECKLIST.md).

## Versioning

For when to bump an eval `task` version, see [TASK_VERSIONING.md](TASK_VERSIONING.md).

## Workflows

For common workflows (reviewing evals, making evaluation reports, checking agent
trajectories, etc.), see [AGENTS.md](AGENTS.md).

## Pull Requests

When creating a pull request, always read `.github/PULL_REQUEST_TEMPLATE.md` and
include its contents in the PR body. Fill in the checklist items and add your
summary above them.

## How to Work

Understand before acting. Read the code, map the dependencies, and understand
why things are the way they are before proposing changes. Present your analysis
and tradeoffs to the user before implementing — let them decide what's worth
changing. Don't start editing files based on assumptions or descriptions you
haven't verified.

## Managed Files

This repo uses a managed file convention. See [MANAGED_FILES.md](MANAGED_FILES.md)
for details on which files are managed (updated from the template) vs.
user-owned.
