"""MMLU/MMMLU correctness eval for the TEE tier x language extension."""

from mmlu.mmlu import get_mmlu_tee_dataset, letter_choice, mmlu_tee

__all__ = ["mmlu_tee", "get_mmlu_tee_dataset", "letter_choice"]
