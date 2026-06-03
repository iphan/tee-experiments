# Total Eval Error — Replication & Extension

A research project **replicating and extending Messing (2026), *Total Eval
Error* (TEE)**. It uses [Inspect AI](https://inspect.aisi.org.uk/) to generate
evaluation data under a factorial of prompt variants and replications, then
decomposes the resulting variance with the
[`totalevalerror`](https://github.com/SolomonMg/totalevalerror) R package.

Built on the
[inspect-eval-template](https://github.com/ArcadiaImpact/inspect-eval-template),
so it supports multiple evaluations in one repository and inherits the template's
quality tooling. MMLU is the first eval implemented here; the repo is designed
as a framework for testing other evaluations against the TEE methodology.

## Background: Total Eval Error

Standard LLM-eval confidence intervals are systematically **too narrow** because
they capture only item-sampling noise and ignore *pipeline* variance — prompt
phrasing, judge choice, temperature, and their interactions. That omitted
variance does **not** shrink as you add items, so reported CIs under-cover worse
as `N` grows. TEE applies generalizability theory to the LLM pipeline: it fits a
linear mixed model over items, prompt variants, replications (and judges, where
present), decomposes the variance into named components, and runs a *design
study* projecting which design changes most reduce error.

The paper demonstrates this only on **English, frontier models**. This project
reproduces the method and extends it to two regimes the paper explicitly flags as
untested:

- **non-frontier models**, and
- **non-English**.

## How it works

```text
Inspect task  (generate + score, per eval)
   │  one sample per (item × prompt variant), run R times via epochs,
   │  temperature > 0, binary correct/incorrect, parse failures flagged
   ▼
.eval logs  ──►  long-format export (item_id, variant_id, epoch, model, language, correct)
   ▼
totalevalerror (R):  tee_design → tee_decompose → tee_se_compare → tee_dstudy
   ▼
per-cell variance tables + cross-cell comparisons + figures
```

Every TEE eval is generated across the same design dimensions:

| Dimension | Meaning | Mechanism |
| --------- | ------- | --------- |
| **N** items | a fixed, frozen item set, reused across cells | dataset rows |
| **V** prompt variants | instruction/framing wrappers; `V_1` is the canonical prompt | one sample per variant |
| **R** replications | repeated calls at temperature > 0 | Inspect **epochs** |
| temperature | > 0, so replicate noise is real | `GenerateConfig` (CLI-overridable) |
| model, language | the extension axes (tier × language) | task params / model flag |

A "cell" is one (model × language) combination; each cell is decomposed
independently and then compared across cells.

## Evaluations

### MMLU / MMMLU correctness — `src/mmlu/` *(implemented)*

Multiple-choice correctness on MMLU (English) and MMMLU (French), with no judge
layer.

- **`src/mmlu/data_gen/`** selects a fixed, stratified, EN/FR-aligned item set
  (100 items, 25 per category across 8 audit-clean subjects) and the five
  hand-authored prompt-variant templates. The selection is frozen and reused
  across all cells; see `src/mmlu/data_gen/data/`.
- **`src/mmlu/mmlu.py`** exposes the `mmlu_tee` task: it reads the selected-items
  CSV, expands each item into one sample per prompt variant, and scores a single
  answer letter (A–D) against the key. Parse failures are recorded distinctly
  (`NOANSWER` + `parse_failed=True`).

Run it (free smoke test with the mock model — no API key, no paid calls):

```bash
uv run inspect eval mmlu_tee \
  -T language=en -T n_items=8 -T n_variants=2 \
  --model mockllm/model --epochs 1
```

A real cell against a provider model:

```bash
# requires OPENROUTER_API_KEY; this makes paid calls
uv run inspect eval mmlu_tee \
  -T language=fr -T n_items=40 -T n_variants=5 \
  --model openrouter/<provider>/<model>
```

Task parameters: `language` (`en`/`fr`), `csv_path`, `n_items` (stratified
subset; `None` = all), `n_variants` (1–5), `seed`. Generation defaults are baked
into the task — `epochs=3`, `temperature=0.7`, `max_tokens=16` — and are
overridable via the standard `--epochs` / `--temperature` / `--max-tokens` flags.

## Adding a new TEE evaluation

Each eval lives in its own directory under `src/` with its own tests under
`tests/<eval>/`. To add one, follow the MMLU pattern:

1. **Freeze an item set.** Select a fixed, representative set of items (a
   `data_gen/` step that writes a CSV is the convention here). Do **not** filter
   items for difficulty or variance — representative sampling is the point.
2. **Author prompt variants.** Write `V ≥ 3` instruction/framing wrappers that
   hold the task, output format, and reasoning mode constant. Include the
   canonical benchmark prompt as `V_1`.
3. **Build the task.** An Inspect `@task` that reads the items, emits one sample
   per (item × variant) with `item_id`, `variant_id`, `language`, and any strata
   in `Sample.metadata`, solves with `generate()`, and scores to binary
   correct/incorrect (flagging parse failures). Bake in `epochs` (R),
   `temperature > 0`, and a small `max_tokens` as defaults.
4. **Wire it up.** Add an `eval.yaml`, export the task from `__init__.py`, and
   register the package under `[project.entry-points.inspect_ai]` in
   `pyproject.toml` (required for eventual `inspect_evals` submission; locally
   you run via the `src/<eval>/<file>.py@<task>` path form).
5. **Test it.** Dataset shape/metadata, scorer behaviour (including parse
   failures), and validation, mirroring `tests/mmlu/`.

See `src/examples/` for template reference implementations (simple Q&A,
LLM-as-judge, GPQA multiple-choice, agentic) — these are not TEE evals, but they
show common Inspect patterns.

## Project structure

```text
src/
  mmlu/                 # MMLU/MMMLU TEE eval (first eval)
    data_gen/           # item selection + prompt-variant authoring (+ frozen data/)
    mmlu.py             # mmlu_tee task, dataset builder, letter scorer
    eval.yaml           # evaluation metadata
  utils/                # shared helpers (stable IDs, metadata, HF loading)
tests/
  mmlu/                 # tests for the MMLU eval
```

The R-side analysis (`totalevalerror`) lives in a separate repository.

## Getting started

```bash
uv sync                                   # install/sync dependencies

# run an eval (mock model = free smoke test)
uv run inspect eval mmlu_tee -T n_items=8 --model mockllm/model --epochs 1

uv run pytest tests/mmlu/                 # run an eval's tests
uv run ruff check && uv run ruff format --check && uv run mypy src tests
```

## Skills

The `.claude/skills/` directory ships Claude Code skills that activate when you
ask Claude to perform the matching task — e.g. "create a new evaluation" or
"review this PR against the template standards". They are managed files, kept up
to date by `sync-template.yml`. The reliable way to invoke one is *"Please run
the /SKILL_NAME skill on EVAL_NAME."*

- **Authoring** — `create-eval`, `investigate-dataset`, `ensure-test-coverage`.
- **Reviewing** — `eval-quality-workflow`, `eval-validity-review`,
  `review-pr-workflow`.
- **Running & analysing** — `eval-report-workflow`, `read-eval-logs`,
  `check-trajectories-workflow`.
- **Submission** — `prepare-submission-workflow`.

## CI workflows

The repo includes GitHub Actions workflows that run automatically:

- **Checks** (`checks.yml`) — ruff, mypy, POSIX code check, unlisted-eval check,
  package build, autolint, generated-docs check, and large-file scan on every
  push and PR. Each check is individually enforceable (see below).
- **Markdown Lint**, **PR Template Check** — markdown style and PR-body
  checklist.
- **Sync Template / Sync Upstream** — weekly sync of managed files from the
  upstream template and from
  [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals).
- **Claude Code Review** (`claude-review.yaml`) — optional Claude-powered PR
  review against the evaluation standards; enabled by adding an
  `ANTHROPIC_API_KEY` (or `ANTHROPIC_ROLE_ARN`) secret.

The sync workflows need **Read and write** workflow permissions, permission to
create PRs, and a `SYNC_PAT` fine-grained token (Contents, Pull requests, and
Workflows = Read and write) to update files under `.github/workflows/`. See
[MANAGED_FILES.md](MANAGED_FILES.md).

## Checks and enforcement

This repo is calibrated against the
[inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals) registry's
quality standards. Those standards are **recommended, not required** — meeting
them smooths the path to registry submission, but they don't block your work.

Each check has an `ENFORCE_<NAME>` setting in
[`tools/enforcement.config`](tools/enforcement.config): `=true` blocks merge on
failure, `=false` reports as advisory. The same file is honoured locally by
`make check` (`tools/run_checks.sh`); environment variables override it for
one-off runs, e.g. `ENFORCE_AUTOLINT=true bash tools/run_checks.sh`.

**Default-enforced:** `ENFORCE_RUFF`, `ENFORCE_MYPY`, `ENFORCE_POSIX_CHECK`,
`ENFORCE_UV_LOCK`, `ENFORCE_UNLISTED_EVALS`, `ENFORCE_PACKAGE`.

**Default-advisory:** `ENFORCE_AUTOLINT`, `ENFORCE_GENERATED_DOCS`,
`ENFORCE_MARKDOWN_LINT`, `ENFORCE_LARGE_FILES`, `ENFORCE_PR_TEMPLATE`.

## Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md) — development setup, testing, and submission
  guidelines.
- [BEST_PRACTICES.md](BEST_PRACTICES.md) — evaluation design best practices
  (synced from `inspect_evals`).
- [EVALUATION_CHECKLIST.md](EVALUATION_CHECKLIST.md) — quality checklist used when
  reviewing evaluations (synced from `inspect_evals`).
- [AUTOMATED_CHECKS.md](AUTOMATED_CHECKS.md) — what `tools/run_autolint.py`
  checks, and how to suppress individual rules.
- [TASK_VERSIONING.md](TASK_VERSIONING.md) — when to bump an eval's `task`
  version.
- [AGENTS.md](AGENTS.md) — repo-wide tips for coding agents and pointers to the
  skills above.
- [MANAGED_FILES.md](MANAGED_FILES.md) — which files are template-managed vs.
  user-owned, and how the sync preserves customizations.
- [CLAUDE.md](CLAUDE.md) — project-level instructions Claude Code reads on every
  session in this repo.
