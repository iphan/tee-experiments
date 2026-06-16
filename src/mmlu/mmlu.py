"""MMLU/MMMLU correctness eval with prompt variants, for the TEE study.

Each selected MMLU item is rendered under the author's original prompt variants
(``mmlu.data_gen.prompt_variants``) and run for several replications (Inspect
epochs), producing the (item x variant x replication) observations the
``totalevalerror`` R pipeline decomposes into variance components.

The full prompt -- instruction wrapper, question, choices, and answer cue -- is
pre-rendered into ``Sample.input`` (no system message, no chain of thought, in
line with the paper's App. O). The model is asked for a single answer letter,
which a small custom scorer extracts and compares to the answer key; unparseable
outputs are recorded distinctly so they can be reported per cell.
"""

from __future__ import annotations

import csv
import io
import random
import re
from importlib import resources
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    NOANSWER,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate

from mmlu.data_gen.prompt_variants import NUM_VARIANTS, VARIANT_IDS, render_prompt
from utils.metadata import load_version_from_yaml

# Paper-correct generation defaults, baked into the Task. Each is overridable via
# the corresponding standard Inspect CLI flag (Task config is the base that the
# CLI args merge over), so they are deliberately not -T task parameters.
DEFAULT_EPOCHS = 3  # R replications; override via --epochs
DEFAULT_TEMPERATURE = 0.7  # >0 needed for genuine replicate variance; --temperature
DEFAULT_MAX_TOKENS = 16  # bare-letter budget; --max-tokens

DATA_PACKAGE = "mmlu.data_gen.data"
EVAL_VERSION = load_version_from_yaml("mmlu")

# Prefer a letter that follows the answer cue ("Answer:" / "Réponse :" /
# "answer is"); allows an optional colon, an "is" gap, markdown bold, and
# surrounding parentheses, e.g. "Answer: (B)", "The correct answer is **a)**".
_CUE_RE = re.compile(
    r"(?i)(?:answer|r[ée]ponse)\b[\s:]*(?:is\b\s*)?\*{0,2}\(?\s*([A-Da-d])\b"
)
# An MCQ choice label: optional markdown/paren, a letter, then ")", e.g.
# "**a)**", "(b)", "c)". Roman numerals ("**I.**") and prose ("An undirected…")
# do not match: they lack a letter immediately followed by ")".
_CHOICE_LABEL_RE = re.compile(r"\*{0,2}\(?\s*([A-Da-d])\)")
# A leading, optionally parenthesised letter, e.g. "A", "a.", "(C)".
_LEADING_RE = re.compile(r"^\(?\s*([A-Da-d])\b")
# Last resort: the first standalone capital letter A-D anywhere (capital only, so
# articles like "a"/"an" are not mistaken for an answer).
_STANDALONE_RE = re.compile(r"\b([A-D])\b")


@task
def mmlu_tee(
    language: Literal["en", "fr"] = "en",
    csv_path: str | None = None,
    n_items: int | None = None,
    n_variants: int = 5,
    seed: int = 42,
) -> Task:
    """MMLU/MMMLU correctness task with prompt variants for the TEE study.

    Args:
        language: ``"en"`` (MMLU) or ``"fr"`` (MMMLU); selects the prompt-variant
            template set and the default items CSV.
        csv_path: Path to a ``selected_items_*`` CSV. Defaults to the packaged
            ``selected_items_{language}.csv``.
        n_items: Use only a stratified subset of this many items (balanced across
            categories/subjects, deterministic given ``seed``). ``None`` uses all
            items in the CSV. This is item-level subsetting, distinct from
            ``--limit`` which truncates the total (item x variant) sample count.
        n_variants: Number of prompt variants to use, 1 to 5 (``v_0``..``v_{n-1}``).
        seed: Seed for the stratified item subset (not the generation seed).

    Returns:
        The configured Inspect ``Task``. Generation defaults (epochs,
        temperature, max_tokens) are overridable via standard CLI flags.
    """
    dataset = get_mmlu_tee_dataset(
        language=language,
        csv_path=csv_path,
        n_items=n_items,
        n_variants=n_variants,
        seed=seed,
    )
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=letter_choice(),
        epochs=DEFAULT_EPOCHS,
        config=GenerateConfig(
            temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_TOKENS
        ),
        version=EVAL_VERSION,
    )


def get_mmlu_tee_dataset(
    language: Literal["en", "fr"] = "en",
    csv_path: str | None = None,
    n_items: int | None = None,
    n_variants: int = 5,
    seed: int = 42,
) -> MemoryDataset:
    """Build the (item x variant) sample set from a selected-items CSV.

    Raises:
        ValueError: if ``n_variants`` is outside 1..5 or ``n_items`` exceeds the
            number of items available in the CSV.
    """
    if not 1 <= n_variants <= NUM_VARIANTS:
        raise ValueError(
            f"n_variants must be between 1 and {NUM_VARIANTS}, got {n_variants}."
        )
    variant_ids = VARIANT_IDS[:n_variants]

    rows = _read_items(csv_path, language)
    if n_items is not None:
        rows = _stratified_subset(rows, n_items, seed)

    samples = [
        Sample(
            input=render_prompt(row, variant_id, language),
            target=row["answer"],
            id=f"{row['item_id']}__{variant_id}",
            metadata={
                "item_id": row["item_id"],
                "variant_id": variant_id,
                "language": language,
                "subject": row["subject"],
                "category": row["category"],
            },
        )
        for row in rows
        for variant_id in variant_ids
    ]
    return MemoryDataset(samples=samples, name=f"mmlu_tee_{language}")


@scorer(metrics=[accuracy(), stderr()])
def letter_choice() -> Scorer:
    """Score a single answer letter against the answer key.

    Extracts an A-D letter from the model output and compares it to the target.
    Outputs with no extractable letter score ``NOANSWER`` with
    ``parse_failed=True`` in the score metadata, keeping parse failures distinct
    from genuine wrong answers so they can be reported (and excluded) per cell.
    """

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        letter = _extract_letter(completion)
        if letter is None:
            return Score(
                value=NOANSWER,
                answer=None,
                metadata={"parse_failed": True, "parsed_letter": None},
                explanation=completion,
            )
        is_correct = letter == target.text.strip().upper()
        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=letter,
            metadata={"parse_failed": False, "parsed_letter": letter},
            explanation=completion,
        )

    return score


def _extract_letter(text: str) -> str | None:
    """Extract an answer letter (A-D) from a model completion, or ``None``."""
    stripped = text.strip()
    for pattern in (_CUE_RE, _CHOICE_LABEL_RE, _LEADING_RE, _STANDALONE_RE):
        match = pattern.search(stripped)
        if match:
            return match.group(1).upper()
    return None


def _read_items(csv_path: str | None, language: str) -> list[dict[str, str]]:
    """Read item rows from a selected-items CSV (default: packaged for ``language``)."""
    if csv_path is not None:
        from pathlib import Path

        text = Path(csv_path).read_text(encoding="utf-8")
    else:
        resource = resources.files(DATA_PACKAGE).joinpath(
            f"selected_items_{language}.csv"
        )
        text = resource.read_text(encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _stratified_subset(
    rows: list[dict[str, str]], n_items: int, seed: int
) -> list[dict[str, str]]:
    """Select ``n_items`` rows balanced across categories then subjects.

    Quotas are split evenly with front-loaded remainders, mirroring the
    selection script's ``allocate_counts``. Sampling within a subject is
    deterministic given ``seed`` (seeded per subject so the result does not
    depend on iteration order). Output is sorted by ``item_id`` for a stable
    dataset order.

    Raises:
        ValueError: if ``n_items`` exceeds the number of rows available.
    """
    if n_items > len(rows):
        raise ValueError(f"n_items ({n_items}) exceeds available items ({len(rows)}).")

    by_category: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in rows:
        by_category.setdefault(row["category"], {}).setdefault(
            row["subject"], []
        ).append(row)

    categories = sorted(by_category)
    selected: list[dict[str, str]] = []
    for category, category_quota in zip(
        categories, _allocate_counts(n_items, len(categories)), strict=True
    ):
        subjects = sorted(by_category[category])
        for subject, subject_quota in zip(
            subjects, _allocate_counts(category_quota, len(subjects)), strict=True
        ):
            subject_rows = sorted(
                by_category[category][subject], key=lambda r: r["item_id"]
            )
            rng = random.Random(f"{seed}-{subject}")
            chosen = rng.sample(subject_rows, min(subject_quota, len(subject_rows)))
            selected.extend(chosen)

    return sorted(selected, key=lambda r: r["item_id"])


def _allocate_counts(total: int, n_buckets: int) -> list[int]:
    """Split ``total`` across ``n_buckets`` as evenly as possible (remainder first)."""
    base, remainder = divmod(total, n_buckets)
    return [base + (1 if i < remainder else 0) for i in range(n_buckets)]
