"""Select a fixed, stratified, EN/FR-aligned set of MMLU items for the TEE study.

Selects ``--n-total`` MMLU items, stratified across broad categories and
subjects, and maps each English item to its French (MMMLU) counterpart so the
same items are evaluated in both languages. The selected set is frozen and
reused across every generation cell, so reproducibility (fixed seed,
recorded in the output) and a verified cross-language alignment are the two
hard requirements.

The EN and FR datasets have no deliberate unique ID; the de-facto join key is
``(subject, position-within-subject)``. To keep that position from depending on
load order or sort stability, the load-order index is captured into ``orig_idx``
before any sort and is included in the sort key.

Run with: ``uv run python src/mmlu/data_gen/select_items.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from mmlu.data_gen.categories import (
    DEFAULT_SUBJECTS_BY_CATEGORY,
    SUBJECT_TO_CATEGORY,
)

logger = logging.getLogger(__name__)

EN_DATASET = "cais/mmlu"
EN_CONFIG = "all"
FR_DATASET = "openai/MMMLU"
FR_CONFIG = "FR_FR"
SPLIT = "test"

ANSWER_LETTERS = "ABCD"
DEFAULT_SEED = 42
DEFAULT_N_TOTAL = 100
ALIGNMENT_CHECK_SAMPLE = 10
ALIGNMENT_CHECK_SEED = 42
EYEBALL_PAIRS = 3

DEFAULT_OUT_DIR = Path(__file__).parent / "data"


def int_to_letter(answer: int) -> str:
    """Map an MMLU integer answer (0-3) to its letter (A-D)."""
    return ANSWER_LETTERS[answer]


def resolve_dataset_revision(path: str, revision: str | None) -> str | None:
    """Resolve a dataset ref to its commit SHA.

    Records the exact dataset version that produced the selection, so the
    frozen item set can be reproduced even after the dataset's default branch
    moves. ``revision`` may be ``None`` (resolves the current default branch),
    a branch, a tag, or a SHA.
    """
    from huggingface_hub import dataset_info

    return dataset_info(path, revision=revision).sha


def allocate_counts(total: int, n_buckets: int) -> list[int]:
    """Split ``total`` across ``n_buckets`` as evenly as possible.

    Remainders are front-loaded, so earlier buckets receive the larger share
    (e.g. ``allocate_counts(25, 2) == [13, 12]``).
    """
    base, remainder = divmod(total, n_buckets)
    return [base + (1 if i < remainder else 0) for i in range(n_buckets)]


def group_subjects_by_category(subjects: list[str]) -> dict[str, list[str]]:
    """Group subjects under their broad category, preserving input order.

    Raises:
        KeyError: if a subject is not present in the embedded MMLU mapping.
    """
    grouped: dict[str, list[str]] = {}
    for subject in subjects:
        category = SUBJECT_TO_CATEGORY[subject]
        grouped.setdefault(category, []).append(subject)
    return grouped


def _add_alignment_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by ``(subject, orig_idx)`` and add ``idx_in_subject`` + ``item_id``."""
    df = df.sort_values(["subject", "orig_idx"]).reset_index(drop=True)
    df["idx_in_subject"] = df.groupby("subject").cumcount()
    df["item_id"] = df["subject"] + "_" + df["idx_in_subject"].astype(str).str.zfill(4)
    return df


def load_mmlu_en(revision: str | None = None) -> pd.DataFrame:
    """Load the English MMLU test split with alignment keys attached.

    Columns: ``subject, orig_idx, idx_in_subject, item_id, question_en,
    choices, answer_en`` (``answer_en`` is the integer 0-3).
    """
    from datasets import load_dataset

    raw = load_dataset(
        EN_DATASET, EN_CONFIG, split=SPLIT, revision=revision
    ).to_pandas()
    df = raw.reset_index(drop=False).rename(
        columns={"index": "orig_idx", "question": "question_en", "answer": "answer_en"}
    )
    df = _add_alignment_keys(df)
    return df[
        [
            "subject",
            "orig_idx",
            "idx_in_subject",
            "item_id",
            "question_en",
            "choices",
            "answer_en",
        ]
    ]


def load_mmmlu_fr(revision: str | None = None) -> pd.DataFrame:
    """Load the French MMMLU test split with alignment keys attached.

    Columns: ``subject, orig_idx, idx_in_subject, item_id, question_fr, A, B,
    C, D, answer_fr`` (``answer_fr`` is the letter A-D).
    """
    from datasets import load_dataset

    raw = load_dataset(
        FR_DATASET, FR_CONFIG, split=SPLIT, revision=revision
    ).to_pandas()
    df = raw.rename(
        columns={
            "Unnamed: 0": "orig_idx",
            "Subject": "subject",
            "Question": "question_fr",
            "Answer": "answer_fr",
        }
    )
    df = _add_alignment_keys(df)
    return df[
        [
            "subject",
            "orig_idx",
            "idx_in_subject",
            "item_id",
            "question_fr",
            "A",
            "B",
            "C",
            "D",
            "answer_fr",
        ]
    ]


def join_en_fr(en: pd.DataFrame, fr: pd.DataFrame) -> pd.DataFrame:
    """Inner-join EN and FR on the positional alignment key."""
    return en.merge(fr, on=["subject", "idx_in_subject", "item_id"], how="inner")


def assert_subject_counts_match(
    en: pd.DataFrame, fr: pd.DataFrame, subjects: list[str]
) -> None:
    """Assert EN and FR have the same row count per subject (alignment proof).

    Raises:
        ValueError: if any subject's EN and FR row counts differ.
    """
    en_counts = en[en["subject"].isin(subjects)].groupby("subject").size()
    fr_counts = fr[fr["subject"].isin(subjects)].groupby("subject").size()
    mismatches = {
        subject: (int(en_counts.get(subject, 0)), int(fr_counts.get(subject, 0)))
        for subject in subjects
        if en_counts.get(subject, 0) != fr_counts.get(subject, 0)
    }
    if mismatches:
        raise ValueError(
            f"EN/FR per-subject row counts differ (subject: (en, fr)): {mismatches}"
        )


def verify_alignment(
    joined: pd.DataFrame,
    n_sample: int = ALIGNMENT_CHECK_SAMPLE,
    seed: int = ALIGNMENT_CHECK_SEED,
) -> int:
    """Spot-check that the EN integer answer matches the FR letter answer.

    Samples ``n_sample`` joined rows and asserts ``ANSWER_LETTERS[answer_en]``
    equals ``answer_fr`` for each. A mismatch means the positional join is not
    aligning the same questions across languages.

    Returns:
        The number of rows checked.

    Raises:
        ValueError: on the first answer mismatch, naming the offending item.
    """
    sample = joined.sample(n=min(n_sample, len(joined)), random_state=seed)
    expected_letter = sample["answer_en"].map(lambda answer: int_to_letter(int(answer)))
    mismatched = sample[expected_letter.to_numpy() != sample["answer_fr"].to_numpy()]
    if not mismatched.empty:
        row = mismatched.iloc[0]
        raise ValueError(
            f"Answer mismatch at {row['subject']}/{row['idx_in_subject']} "
            f"(item {row['item_id']}): EN={expected_letter.loc[mismatched.index[0]]} "
            f"FR={row['answer_fr']} -- EN/FR alignment broken"
        )
    return len(sample)


def log_eyeball_pairs(
    items: pd.DataFrame,
    n_pairs: int = EYEBALL_PAIRS,
    seed: int = ALIGNMENT_CHECK_SEED,
) -> None:
    """Log up to ``n_pairs`` EN/FR question pairs per subject for verification."""
    sample = pd.concat(
        group.sample(n=min(n_pairs, len(group)), random_state=seed)
        for _, group in items.groupby("subject")
    )
    logger.info(f"Verify the following {n_pairs} pair(s) per subject for matching:")
    for row in sample.itertuples():
        logger.info("  EN [%s]: %s", row.item_id, row.question_en)
        logger.info("  FR [%s]: %s\n", row.item_id, row.question_fr)


def stratified_sample(
    joined: pd.DataFrame,
    subjects_by_category: dict[str, list[str]],
    n_total: int,
    seed: int,
) -> pd.DataFrame:
    """Draw a stratified sample: equal per category, even split across subjects.

    Each category receives ``n_total / n_categories`` items, split across its
    subjects via ``allocate_counts``. Sampling within a subject uses ``seed``.
    """
    per_category = allocate_counts(n_total, len(subjects_by_category))
    parts: list[pd.DataFrame] = []
    for (category, subjects), category_quota in zip(
        subjects_by_category.items(), per_category, strict=True
    ):
        subject_quotas = allocate_counts(category_quota, len(subjects))
        for subject, quota in zip(subjects, subject_quotas, strict=True):
            subject_rows = joined[joined["subject"] == subject]
            parts.append(subject_rows.sample(n=quota, random_state=seed))
    return pd.concat(parts).sort_values("item_id").reset_index(drop=True)


def build_output_frames(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the EN and FR output frames with the shared output schema."""
    common = sample[["item_id", "subject", "category", "idx_in_subject"]].reset_index(
        drop=True
    )

    en = common.copy()
    en["question"] = sample["question_en"].to_numpy()
    for offset, letter in enumerate(ANSWER_LETTERS):
        en[f"choice_{letter}"] = [choices[offset] for choices in sample["choices"]]
    en["answer"] = [int_to_letter(int(answer)) for answer in sample["answer_en"]]

    fr = common.copy()
    fr["question"] = sample["question_fr"].to_numpy()
    for letter in ANSWER_LETTERS:
        fr[f"choice_{letter}"] = sample[letter].to_numpy()
    fr["answer"] = sample["answer_fr"].to_numpy()

    return en, fr


def write_outputs(
    out_dir: Path,
    en: pd.DataFrame,
    fr: pd.DataFrame,
    meta: dict[str, object],
) -> None:
    """Write the item-id list, both CSVs, and the selection metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    item_ids = sorted(en["item_id"])
    (out_dir / "selected_item_ids.txt").write_text("\n".join(item_ids) + "\n")
    en.to_csv(out_dir / "selected_items_en.csv", index=False)
    fr.to_csv(out_dir / "selected_items_fr.csv", index=False)
    (out_dir / "selection_meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def select_items(
    subjects_by_category: dict[str, list[str]],
    n_total: int,
    seed: int,
    en_revision: str | None,
    fr_revision: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run the full selection pipeline, returning EN/FR frames and metadata."""
    subjects = [s for subjects in subjects_by_category.values() for s in subjects]

    en = load_mmlu_en(en_revision)
    fr = load_mmmlu_fr(fr_revision)
    assert_subject_counts_match(en, fr, subjects)

    joined = join_en_fr(en, fr)
    n_checked = verify_alignment(joined)

    chosen = joined[joined["subject"].isin(subjects)].copy()
    chosen["category"] = chosen["subject"].map(SUBJECT_TO_CATEGORY)
    sample = stratified_sample(chosen, subjects_by_category, n_total, seed)
    log_eyeball_pairs(sample)

    en_out, fr_out = build_output_frames(sample)

    per_subject = sample.groupby("subject").size().to_dict()
    per_category = sample.groupby("category").size().to_dict()
    meta: dict[str, object] = {
        "seed": seed,
        "n_total": n_total,
        "subjects_by_category": subjects_by_category,
        "per_subject_counts": {k: int(v) for k, v in per_subject.items()},
        "per_category_counts": {k: int(v) for k, v in per_category.items()},
        "alignment_check": {"rows_checked": n_checked, "passed": True},
        "datasets": {
            "en": {
                "path": EN_DATASET,
                "config": EN_CONFIG,
                "split": SPLIT,
                "revision": resolve_dataset_revision(EN_DATASET, en_revision),
            },
            "fr": {
                "path": FR_DATASET,
                "config": FR_CONFIG,
                "split": SPLIT,
                "revision": resolve_dataset_revision(FR_DATASET, fr_revision),
            },
        },
    }
    return en_out, fr_out, meta


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-total", type=int, default=DEFAULT_N_TOTAL)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Subjects to sample from (default: the 8 study subjects).",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--en-revision", default=None, help="Pin cais/mmlu revision.")
    parser.add_argument(
        "--fr-revision", default=None, help="Pin openai/MMMLU revision."
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: select items and write the output files."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    subjects_by_category = (
        group_subjects_by_category(args.subjects)
        if args.subjects is not None
        else DEFAULT_SUBJECTS_BY_CATEGORY
    )

    en, fr, meta = select_items(
        subjects_by_category=subjects_by_category,
        n_total=args.n_total,
        seed=args.seed,
        en_revision=args.en_revision,
        fr_revision=args.fr_revision,
    )
    write_outputs(args.out_dir, en, fr, meta)

    logger.info("Seed: %s", meta["seed"])
    logger.info("Per-category counts: %s", meta["per_category_counts"])
    logger.info("Per-subject counts: %s", meta["per_subject_counts"])
    logger.info("Alignment check: %s", meta["alignment_check"])
    logger.info("Wrote %s items to %s", len(en), args.out_dir)


if __name__ == "__main__":
    main()
