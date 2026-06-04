"""Tests for the MMLU TEE Inspect task (dataset construction and scorer)."""

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER
from inspect_ai.solver import generate

from mmlu.data_gen.prompt_variants import NUM_VARIANTS, VARIANT_IDS
from mmlu.mmlu import (
    _extract_letter,
    get_mmlu_tee_dataset,
    letter_choice,
    mmlu_tee,
)

STUDY_SUBJECTS = {
    "college_computer_science",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "marketing",
    "nutrition",
    "philosophy",
    "sociology",
    "world_religions",
}


def _item_ids(dataset: MemoryDataset) -> list[str]:
    """Distinct item_ids in a dataset, in first-seen order."""
    seen: dict[str, None] = {}
    for sample in dataset:
        assert sample.metadata is not None
        seen.setdefault(sample.metadata["item_id"], None)
    return list(seen)


def _score_one(target: str, content: str) -> tuple[object, dict[str, object]]:
    """Run the scorer end-to-end on a single synthetic sample via mockllm.

    Returns the score value and its metadata.
    """
    sut = Task(
        dataset=MemoryDataset([Sample(input="q", target=target, id="item__v_0")]),
        solver=generate(),
        scorer=letter_choice(),
    )
    [log] = eval(
        sut,
        model=get_model(
            "mockllm/model",
            custom_outputs=[
                ModelOutput.from_content(model="mockllm/model", content=content)
            ],
        ),
        epochs=1,
    )
    assert log.samples is not None
    assert log.samples[0].scores is not None
    score = next(iter(log.samples[0].scores.values()))
    return score.value, score.metadata or {}


class TestDatasetShape:
    def test_sample_count_is_items_times_variants(self) -> None:
        # given a stratified subset of items and a variant count
        # when the dataset is built
        sut = get_mmlu_tee_dataset(n_items=8, n_variants=3)
        # then there is one sample per (item, variant)
        assert len(sut) == 8 * 3
        assert len(_item_ids(sut)) == 8

    def test_default_uses_all_items_and_five_variants(self) -> None:
        # given no subset / variant overrides (the full English set has 100 items)
        # when the dataset is built
        sut = get_mmlu_tee_dataset()
        # then all items appear under all five variants
        assert len(sut) == 100 * NUM_VARIANTS
        assert len(_item_ids(sut)) == 100

    def test_sample_id_embeds_item_and_variant(self) -> None:
        # given a built dataset
        sut = get_mmlu_tee_dataset(n_items=4, n_variants=2)
        # when each sample id is parsed
        for sample in sut:
            assert sample.metadata is not None
            item_id = sample.metadata["item_id"]
            variant_id = sample.metadata["variant_id"]
            # then the id is exactly "{item_id}__{variant_id}"
            assert sample.id == f"{item_id}__{variant_id}"
            assert variant_id in VARIANT_IDS[:2]

    def test_metadata_carries_export_fields(self) -> None:
        # given a built dataset / when a sample is inspected
        sut = get_mmlu_tee_dataset(n_items=4, n_variants=1)
        sample = next(iter(sut))
        # then it exposes the keys the long-format export reads
        assert sample.metadata is not None
        assert set(sample.metadata) == {
            "item_id",
            "variant_id",
            "language",
            "subject",
            "category",
        }
        assert sample.metadata["language"] == "en"

    def test_target_is_an_answer_letter(self) -> None:
        # given a built dataset / when targets are inspected
        sut = get_mmlu_tee_dataset(n_items=4, n_variants=1)
        # then every target is a single A-D letter
        assert all(sample.target in {"A", "B", "C", "D"} for sample in sut)


class TestLanguage:
    def test_french_prompt_uses_french_wording(self) -> None:
        # given the French language
        # when a sample prompt is rendered (the canonical v_0 variant)
        sut = get_mmlu_tee_dataset(language="fr", n_items=4, n_variants=1)
        sample = next(iter(sut))
        # then it carries French instruction wording and the language tag
        assert "Répondez à la question à choix multiples" in str(sample.input)
        assert sample.metadata is not None
        assert sample.metadata["language"] == "fr"

    def test_english_prompt_uses_english_wording(self) -> None:
        # given the English language / when a prompt is rendered (canonical v_0)
        sut = get_mmlu_tee_dataset(language="en", n_items=4, n_variants=1)
        sample = next(iter(sut))
        # then it carries English instruction wording
        assert "Answer the following multiple-choice question" in str(sample.input)


class TestValidation:
    @pytest.mark.parametrize("n_variants", [0, NUM_VARIANTS + 1, -1])
    def test_out_of_range_variants_raise(self, n_variants: int) -> None:
        # given an out-of-range variant count / when the dataset is built
        # then a ValueError is raised
        with pytest.raises(ValueError, match="n_variants"):
            get_mmlu_tee_dataset(n_items=4, n_variants=n_variants)

    def test_too_many_items_raises(self) -> None:
        # given more items requested than the CSV holds
        # when the dataset is built / then a ValueError is raised
        with pytest.raises(ValueError, match="exceeds available"):
            get_mmlu_tee_dataset(n_items=10_000)


class TestStratifiedSubset:
    def test_small_subset_is_balanced_across_subjects(self) -> None:
        # given a subset of exactly one item per study subject
        sut = get_mmlu_tee_dataset(n_items=8, n_variants=1)
        # when the chosen subjects are collected
        subjects = {
            sample.metadata["subject"] for sample in sut if sample.metadata is not None
        }
        # then all eight subjects are represented
        assert subjects == STUDY_SUBJECTS

    def test_subset_is_deterministic_given_seed(self) -> None:
        # given two builds with the same seed
        first = _item_ids(get_mmlu_tee_dataset(n_items=12, n_variants=1, seed=7))
        second = _item_ids(get_mmlu_tee_dataset(n_items=12, n_variants=1, seed=7))
        # when their item ids are compared / then they match exactly
        assert first == second


class TestExtractLetter:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("A", "A"),
            ("a", "A"),
            ("(C)", "C"),
            ("B.", "B"),
            ("Answer: D", "D"),
            ("Réponse : C", "C"),
            ("The answer is C.", "C"),
        ],
    )
    def test_parses_letter(self, text: str, expected: str) -> None:
        # given a model output / when a letter is extracted / then it is correct
        assert _extract_letter(text) == expected

    @pytest.mark.parametrize("text", ["", "I am not sure", "let me think"])
    def test_unparseable_returns_none(self, text: str) -> None:
        # given output with no answer letter / when extracted / then None
        assert _extract_letter(text) is None


class TestScorer:
    def test_matching_letter_scores_correct(self) -> None:
        # given a model output matching the target / when scored
        value, metadata = _score_one(target="B", content="B")
        # then the score is CORRECT and not a parse failure
        assert value == CORRECT
        assert metadata["parse_failed"] is False
        assert metadata["parsed_letter"] == "B"

    def test_wrong_letter_scores_incorrect(self) -> None:
        # given a parseable but wrong answer / when scored / then INCORRECT
        value, metadata = _score_one(target="B", content="A")
        assert value == INCORRECT
        assert metadata["parse_failed"] is False

    def test_unparseable_scores_noanswer_and_flags(self) -> None:
        # given an output with no extractable letter / when scored
        value, metadata = _score_one(target="B", content="I really cannot tell")
        # then it is NOANSWER and flagged so the export can exclude it
        assert value == NOANSWER
        assert metadata["parse_failed"] is True


class TestTaskStructure:
    def test_task_has_baked_in_defaults(self) -> None:
        # given the task constructed with defaults / when inspected
        sut = mmlu_tee(n_items=4, n_variants=1)
        # then epochs and generation config carry the paper values
        assert sut.epochs == 3
        assert sut.config.temperature == 0.7
        assert sut.config.max_tokens == 16
        assert sut.scorer is not None
