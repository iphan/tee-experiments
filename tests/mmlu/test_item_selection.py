"""Tests for the MMLU/MMMLU item-selection script."""

import pandas as pd
import pytest

from mmlu.data_gen.categories import (
    DEFAULT_SUBJECTS_BY_CATEGORY,
    SUBJECT_TO_CATEGORY,
)
from mmlu.data_gen.select_items import (
    allocate_counts,
    assert_subject_counts_match,
    build_output_frames,
    group_subjects_by_category,
    int_to_letter,
    join_en_fr,
    select_items,
    stratified_sample,
    verify_alignment,
)
from utils.huggingface import (
    DatasetInfosDict,
    assert_huggingface_dataset_structure,
    get_dataset_infos_dict,
)

OUTPUT_COLUMNS = [
    "item_id",
    "subject",
    "category",
    "idx_in_subject",
    "question",
    "choice_A",
    "choice_B",
    "choice_C",
    "choice_D",
    "answer",
]


class TestPureLogic:
    def test_int_to_letter(self) -> None:
        # given / when / then
        assert [int_to_letter(i) for i in range(4)] == ["A", "B", "C", "D"]

    def test_allocate_counts_even(self) -> None:
        # given a total that divides evenly / when / then
        assert allocate_counts(100, 4) == [25, 25, 25, 25]

    def test_allocate_counts_front_loads_remainder(self) -> None:
        # given a per-category quota of 25 across 2 subjects
        # when / then the first subject takes the larger share
        assert allocate_counts(25, 2) == [13, 12]

    def test_default_subjects_resolve_to_their_category(self) -> None:
        # given the default by-category grouping
        # when / then every listed subject maps back to that category
        for category, subjects in DEFAULT_SUBJECTS_BY_CATEGORY.items():
            for subject in subjects:
                assert SUBJECT_TO_CATEGORY[subject] == category

    def test_group_subjects_by_category(self) -> None:
        # given a flat subject list spanning two categories
        subjects = ["marketing", "philosophy", "nutrition"]
        # when
        grouped = group_subjects_by_category(subjects)
        # then
        assert grouped == {
            "Other": ["marketing", "nutrition"],
            "Humanities": ["philosophy"],
        }

    def test_group_subjects_rejects_unknown(self) -> None:
        # given an unknown subject / when / then
        with pytest.raises(KeyError):
            group_subjects_by_category(["not_a_subject"])


def _synthetic_side(language: str, rows_per_subject: int = 20) -> pd.DataFrame:
    """Build a synthetic per-language frame already carrying alignment keys."""
    records = []
    for subject in ["marketing", "philosophy"]:
        for idx in range(rows_per_subject):
            item_id = f"{subject}_{idx:04d}"
            answer_int = idx % 4
            record = {
                "subject": subject,
                "idx_in_subject": idx,
                "item_id": item_id,
            }
            if language == "en":
                record |= {
                    "question_en": f"EN {item_id}",
                    "choices": [f"{item_id}-{letter}" for letter in "ABCD"],
                    "answer_en": answer_int,
                }
            else:
                record |= {
                    "question_fr": f"FR {item_id}",
                    "A": f"{item_id}-A",
                    "B": f"{item_id}-B",
                    "C": f"{item_id}-C",
                    "D": f"{item_id}-D",
                    "answer_fr": "ABCD"[answer_int],
                }
            records.append(record)
    return pd.DataFrame(records)


class TestAlignmentAndSampling:
    def test_subject_counts_match(self) -> None:
        # given aligned EN/FR frames / when / then no error
        en, fr = _synthetic_side("en"), _synthetic_side("fr")
        assert_subject_counts_match(en, fr, ["marketing", "philosophy"])

    def test_subject_counts_mismatch_raises(self) -> None:
        # given FR with fewer rows for one subject
        en = _synthetic_side("en")
        fr = _synthetic_side("fr")
        fr = fr.drop(fr.index[-1])
        # when / then
        with pytest.raises(ValueError, match="per-subject row counts differ"):
            assert_subject_counts_match(en, fr, ["marketing", "philosophy"])

    def test_verify_alignment_passes_when_answers_agree(self) -> None:
        # given a correctly aligned join / when / then
        joined = join_en_fr(_synthetic_side("en"), _synthetic_side("fr"))
        assert verify_alignment(joined) == 10

    def test_verify_alignment_raises_on_answer_mismatch(self) -> None:
        # given a join whose FR answers were corrupted
        fr = _synthetic_side("fr")
        fr["answer_fr"] = "Z"
        joined = join_en_fr(_synthetic_side("en"), fr)
        # when / then
        with pytest.raises(ValueError, match="alignment broken"):
            verify_alignment(joined)

    def test_stratified_sample_balances_categories_and_subjects(self) -> None:
        # given a join over the default subjects
        joined = join_en_fr(_synthetic_side("en"), _synthetic_side("fr"))
        joined["category"] = joined["subject"].map(SUBJECT_TO_CATEGORY)
        subjects_by_category = {"Other": ["marketing", "philosophy"]}
        # when sampling 25 from one category of 2 subjects
        sample = stratified_sample(joined, subjects_by_category, n_total=25, seed=42)
        # then totals and the 13/12 split hold
        assert len(sample) == 25
        counts = sample.groupby("subject").size().to_dict()
        assert counts == {"marketing": 13, "philosophy": 12}

    def test_build_output_frames_schema_and_answers(self) -> None:
        # given a small sampled frame with a category column
        joined = join_en_fr(_synthetic_side("en"), _synthetic_side("fr"))
        joined["category"] = joined["subject"].map(SUBJECT_TO_CATEGORY)
        # when
        en, fr = build_output_frames(joined)
        # then both frames share the output schema
        assert list(en.columns) == OUTPUT_COLUMNS
        assert list(fr.columns) == OUTPUT_COLUMNS
        # and the EN integer answer is rendered as the matching letter
        first = joined.iloc[0]
        assert en.iloc[0]["answer"] == int_to_letter(int(first["answer_en"]))
        assert fr.iloc[0]["answer"] == first["answer_fr"]


@pytest.fixture(scope="module")
def en_infos() -> DatasetInfosDict:
    return get_dataset_infos_dict("cais/mmlu")


@pytest.fixture(scope="module")
def fr_infos() -> DatasetInfosDict:
    return get_dataset_infos_dict("openai/MMMLU")


@pytest.mark.huggingface
def test_en_dataset_structure(en_infos: DatasetInfosDict) -> None:
    assert_huggingface_dataset_structure(
        en_infos,
        {
            "configs": {
                "all": {
                    "splits": ["test"],
                    "features": {
                        "question": "string",
                        "subject": "string",
                        "choices": "List",
                        "answer": "ClassLabel",
                    },
                }
            }
        },
    )


@pytest.mark.huggingface
def test_fr_dataset_structure(fr_infos: DatasetInfosDict) -> None:
    assert_huggingface_dataset_structure(
        fr_infos,
        {
            "configs": {
                "FR_FR": {
                    "splits": ["test"],
                    "features": {
                        "Question": "string",
                        "Subject": "string",
                        "Answer": "string",
                        "A": "string",
                    },
                }
            }
        },
    )


@pytest.mark.dataset_download
def test_select_items_end_to_end() -> None:
    # given the default 8-subject design / when running the full pipeline
    en, fr, meta = select_items(
        subjects_by_category=DEFAULT_SUBJECTS_BY_CATEGORY,
        n_total=100,
        seed=42,
        en_revision=None,
        fr_revision=None,
    )
    # then 100 items, balanced 25/category and 12-13/subject
    assert len(en) == 100
    assert set(en["item_id"]) == set(fr["item_id"])
    assert meta["per_category_counts"] == {
        "STEM": 25,
        "Humanities": 25,
        "Social Sciences": 25,
        "Other": 25,
    }
    assert all(12 <= count <= 13 for count in meta["per_subject_counts"].values())
    # and the exact dataset commit SHAs are recorded for reproducibility
    assert meta["datasets"]["en"]["revision"]
    assert meta["datasets"]["fr"]["revision"]
