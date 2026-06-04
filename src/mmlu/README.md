# MMLU TEE: Multiple-Choice Correctness Under Prompt Variants

Each selected MMLU item is rendered under the author's original prompt variants and run for several replications (Inspect
epochs), producing the (item x variant x replication) observations the ``totalevalerror`` R pipeline decomposes into variance components.

<!-- Contributors: Automatically Generated -->
Contributed by [@iphan](https://github.com/iphan)
<!-- /Contributors: Automatically Generated -->

<!-- Usage: Automatically Generated -->
## Usage

First, install dependencies:

```bash
uv sync
```

Then run evaluations:

```bash
uv run inspect eval mmlu/mmlu_tee --model openai/gpt-5-nano
```

You can also import tasks as Python objects:

```python
from inspect_ai import eval
from mmlu import mmlu_tee
eval(mmlu_tee)
```

After running evaluations, view logs with:

```bash
uv run inspect view
```

If you don't want to specify `--model` each time, create a `.env` file:

```bash
INSPECT_EVAL_MODEL=anthropic/claude-opus-4-1-20250805
ANTHROPIC_API_KEY=<anthropic-api-key>
```
<!-- /Usage: Automatically Generated -->

<!-- Options: Automatically Generated -->
## Options

You can control a variety of options from the command line. For example:

```bash
uv run inspect eval mmlu/mmlu_tee --limit 10
uv run inspect eval mmlu/mmlu_tee --max-connections 10
uv run inspect eval mmlu/mmlu_tee --temperature 0.5
```

See `uv run inspect eval --help` for all available options.
<!-- /Options: Automatically Generated -->

<!-- Parameters: Automatically Generated -->
## Parameters

### `mmlu_tee`

- `language` (Literal['en', 'fr']): ``"en"`` (MMLU) or ``"fr"`` (MMMLU); selects the prompt-variant template set and the default items CSV. (default: `'en'`)
- `csv_path` (str | None): Path to a ``selected_items_*`` CSV. Defaults to the packaged ``selected_items_{language}.csv``. (default: `None`)
- `n_items` (int | None): Use only a stratified subset of this many items (balanced across categories/subjects, deterministic given ``seed``). ``None`` uses all items in the CSV. This is item-level subsetting, distinct from ``--limit`` which truncates the total (item x variant) sample count. (default: `None`)
- `n_variants` (int): Number of prompt variants to use, 1 to 5 (``v_0``..``v_{n-1}``). (default: `5`)
- `seed` (int): Seed for the stratified item subset (not the generation seed). (default: `42`)
<!-- /Parameters: Automatically Generated -->

Example usage

```bash
# all 100 items x 5 variants x 3 epochs, English
uv run inspect eval mmlu_tee --model openai/gpt-5-nano

# a smaller French cell
uv run inspect eval mmlu_tee --model openai/gpt-5-nano -T language=fr -T n_items=40 -T n_variants=5

# override the baked-in generation defaults via standard CLI flags
uv run inspect eval mmlu_tee --model openai/gpt-5-nano --temperature 1.0 --max-tokens 32 --epochs 5
```

## Dataset

100 items from MMLU and MMMLU (French) were selected across 8 subjects:

- STEM: high_school_mathematics, college_computer_science
- Humanities: world_religions, philosophy
- Social Sciences: sociology, high_school_macroeconomics
- Other: marketing, nutrition

## Prompt variants

Each item is rendered under five prompt variants — Messing (2026)'s originals,
vendored verbatim in [`data_gen/prompts/mmlu_variants.md`](data_gen/prompts/mmlu_variants.md)
(French translation in [`mmlu_variants_fr.md`](data_gen/prompts/mmlu_variants_fr.md)).
Every variant wraps the same question and choices; only the instruction framing
and the answer-option delimiter vary. The task, output format (a single A-D
letter), and reasoning mode (direct answer) are held constant (Assumption 1).
`v_0` is the canonical reference variant.

| ID    | Framing                          | Option delimiter |
| ----- | -------------------------------- | ---------------- |
| `v_0` | Standard                         | `A.`             |
| `v_1` | Exam                             | `(A)`            |
| `v_2` | Expert                           | `A)`             |
| `v_3` | Minimal (trailing `Answer:` cue) | `A.`             |
| `v_4` | Analytical                       | lowercase `a)`   |

`n_variants` selects the first `n` in order, so a reduced run always includes
the canonical `v_0` first.

## Scoring

The scorer extracts an A-D letter from the model output and compares it to the target.
Outputs with no extractable letter score ``NOANSWER`` with ``parse_failed=True`` in the score metadata, keeping parse failures distinct from genuine wrong answers so they can be reported (and excluded) per cell.

## Evaluation Report

TODO: A brief summary of results for your evaluation implementation compared against a standard set of existing results.

## Changelog
