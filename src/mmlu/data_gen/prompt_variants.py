"""The author's original MMLU/MMMLU prompt variants and the item-wrapping helper.

In Total Eval Error (Messing 2026) the MMLU "prompt variant" is the instruction
wrapper around a question, not a rewording of the question itself. These are the
**author's original five variants** (published after correspondence), vendored
verbatim under ``prompts/mmlu_variants.md``; ``prompts/mmlu_variants_fr.md`` holds
our French translation. Each variant differs only in instruction phrasing,
framing, and the answer-option delimiter (``A.``, ``(A)``, ``A)``, lowercase
``a)``) while holding constant the task and the verbatim question and choices.
``v_0`` ("Standard") is the canonical reference variant.

The choices are embedded inline in every template, so unlike the canonical MMLU
prompt the variants carry no subject and no single fixed answer cue (only the
"Minimal" variant ``v_3`` ends with an ``Answer:`` line).
"""

from __future__ import annotations

from collections.abc import Mapping

ANSWER_LETTERS = "ABCD"
NUM_VARIANTS = 5
VARIANT_IDS = [f"v_{i}" for i in range(NUM_VARIANTS)]
ORIGINAL_VARIANT_ID = "v_0"
LANGUAGES = ("en", "fr")

# Each template fills {question} and {A}..{D} (the choice texts). The instruction
# phrasing, framing, and option delimiter vary per variant; v_0 is canonical.
# Mirrors prompts/mmlu_variants.md verbatim.
EN_TEMPLATES = [
    "Answer the following multiple-choice question. Reply with only the letter "
    "(A, B, C, or D).\n"
    "\nQuestion: {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}",
    "You are taking a test. Select the correct answer for the question below. "
    "Respond with just the letter.\n"
    "\nQ: {question}\n(A) {A}\n(B) {B}\n(C) {C}\n(D) {D}",
    "As a knowledgeable expert, identify the correct answer to this question. "
    "Output only the letter of the correct choice.\n"
    "\n{question}\nOptions:\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
    "{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\nAnswer:",
    "Read the following question carefully and select the best answer from the "
    "options provided. State only the letter.\n"
    "\nQuestion: {question}\nChoices:\na) {A}\nb) {B}\nc) {C}\nd) {D}",
]

# French translation (prompts/mmlu_variants_fr.md): instruction prose and labels
# are translated; the option delimiters and template structure match EN.
FR_TEMPLATES = [
    "Répondez à la question à choix multiples suivante. Indiquez uniquement la "
    "lettre (A, B, C ou D).\n"
    "\nQuestion : {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}",
    "Vous passez un examen. Sélectionnez la bonne réponse à la question "
    "ci-dessous. Répondez uniquement par la lettre.\n"
    "\nQ : {question}\n(A) {A}\n(B) {B}\n(C) {C}\n(D) {D}",
    "En tant qu'expert compétent, identifiez la bonne réponse à cette question. "
    "Indiquez uniquement la lettre du bon choix.\n"
    "\n{question}\nOptions :\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
    "{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\nRéponse :",
    "Lisez attentivement la question suivante et choisissez la meilleure réponse "
    "parmi les options proposées. Indiquez uniquement la lettre.\n"
    "\nQuestion : {question}\nChoix :\na) {A}\nb) {B}\nc) {C}\nd) {D}",
]

TEMPLATES: dict[str, list[str]] = {"en": EN_TEMPLATES, "fr": FR_TEMPLATES}


def render_prompt(row: Mapping[str, object], variant_id: str, language: str) -> str:
    """Render one item under one prompt variant.

    Args:
        row: An item row exposing ``question`` and ``choice_A``..``choice_D``
            (e.g. a record from ``selected_items_{en,fr}.csv``).
        variant_id: One of ``VARIANT_IDS`` (``v_0``..``v_4``).
        language: ``"en"`` or ``"fr"``; selects the template set.

    Returns:
        The fully rendered user message sent to the system under test.

    Raises:
        ValueError: if ``language`` or ``variant_id`` is unknown.
    """
    if language not in TEMPLATES:
        raise ValueError(f"Unknown language {language!r}; expected one of {LANGUAGES}.")
    if variant_id not in VARIANT_IDS:
        raise ValueError(
            f"Unknown variant_id {variant_id!r}; expected one of {VARIANT_IDS}."
        )
    template = TEMPLATES[language][VARIANT_IDS.index(variant_id)]
    choices = {letter: row[f"choice_{letter}"] for letter in ANSWER_LETTERS}
    return template.format(question=row["question"], **choices)
