"""Tests for the author's original MMLU/MMMLU prompt variants."""

import pytest

from mmlu.data_gen.prompt_variants import (
    ANSWER_LETTERS,
    EN_TEMPLATES,
    FR_TEMPLATES,
    NUM_VARIANTS,
    ORIGINAL_VARIANT_ID,
    VARIANT_IDS,
    render_prompt,
)

PLACEHOLDERS = ["{question}", "{A}", "{B}", "{C}", "{D}"]

# The canonical variant (v_0, "Standard"): inline choices, no subject.
CANONICAL_EN_V0 = (
    "Answer the following multiple-choice question. Reply with only the letter "
    "(A, B, C, or D).\n"
    "\nQuestion: Which is true?\nA. first\nB. second\nC. third\nD. fourth"
)


def _synthetic_row() -> dict[str, object]:
    """Build a synthetic item row with the selected-items schema."""
    return {
        "question": "Which is true?",
        "choice_A": "first",
        "choice_B": "second",
        "choice_C": "third",
        "choice_D": "fourth",
        "answer": "A",
    }


class TestConstants:
    def test_five_index_aligned_templates_per_language(self) -> None:
        # given / when / then both languages expose exactly NUM_VARIANTS templates
        assert len(EN_TEMPLATES) == NUM_VARIANTS
        assert len(FR_TEMPLATES) == NUM_VARIANTS
        assert len(VARIANT_IDS) == NUM_VARIANTS

    def test_original_is_a_known_variant(self) -> None:
        # given / when / then the original is pinned to a real variant id
        assert ORIGINAL_VARIANT_ID == "v_0"
        assert ORIGINAL_VARIANT_ID in VARIANT_IDS

    def test_every_template_has_all_placeholders(self) -> None:
        # given every template / when / then all fill fields are present
        for template in EN_TEMPLATES + FR_TEMPLATES:
            for placeholder in PLACEHOLDERS:
                assert placeholder in template


class TestRenderPrompt:
    @pytest.mark.parametrize("language", ["en", "fr"])
    @pytest.mark.parametrize("variant_id", VARIANT_IDS)
    def test_fills_every_placeholder_and_keeps_content_verbatim(
        self, language: str, variant_id: str
    ) -> None:
        # given a synthetic row / when rendering each variant in each language
        row = _synthetic_row()
        prompt = render_prompt(row, variant_id, language)
        # then no placeholder is left unfilled
        for placeholder in PLACEHOLDERS:
            assert placeholder not in prompt
        # and the question and every choice survive verbatim (item is unchanged)
        assert str(row["question"]) in prompt
        for letter in ANSWER_LETTERS:
            assert str(row[f"choice_{letter}"]) in prompt

    @pytest.mark.parametrize("templates", [EN_TEMPLATES, FR_TEMPLATES])
    def test_variants_are_distinct(self, templates: list[str]) -> None:
        # given a single language's templates / when / then all five differ
        assert len(set(templates)) == NUM_VARIANTS

    def test_v0_matches_canonical_standard_prompt(self) -> None:
        # given the canonical variant / when / then it is the author's Standard format
        assert render_prompt(_synthetic_row(), "v_0", "en") == CANONICAL_EN_V0

    def test_rejects_unknown_variant(self) -> None:
        # given an unknown variant id / when / then
        with pytest.raises(ValueError, match="Unknown variant_id"):
            render_prompt(_synthetic_row(), "v_99", "en")

    def test_rejects_unknown_language(self) -> None:
        # given an unknown language / when / then
        with pytest.raises(ValueError, match="Unknown language"):
            render_prompt(_synthetic_row(), "v_0", "de")
