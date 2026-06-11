"""Unit tests for ``bench_interpretation._extract_letter``.

The reviewer of FlowAgent's interpretation benchmark observed that
``_extract_letter`` returned ``I`` for replies like "I believe the
answer is B". The fix replaces the regex with a tag-aware parser
plus a tiered fallback. These tests pin every regression case the
reviewer flagged, and a few defensive edge cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(BENCH_DIR.parent))

from bench_interpretation import _extract_letter  # noqa: E402


VALID = ["A", "B", "C", "D"]


class TestStructuredTag:
    def test_tag_only(self):
        assert _extract_letter("<answer>B</answer>", VALID) == "B"

    def test_tag_with_explain(self):
        assert _extract_letter(
            "<answer>C</answer><explain>because of x</explain>", VALID,
        ) == "C"

    def test_tag_with_surrounding_prose(self):
        assert _extract_letter(
            "Sure! Here is my reply.\n<answer>D</answer>\n<explain>..</explain>",
            VALID,
        ) == "D"

    def test_tag_lowercase(self):
        assert _extract_letter("<answer>b</answer>", VALID) == "B"


class TestReviewerRegressionCases:
    def test_first_letter_I_does_not_leak(self):
        # The bug the reviewer flagged.
        assert _extract_letter("I believe the answer is B.", VALID) == "B"

    def test_apostrophe_lead_then_answer(self):
        assert _extract_letter("It's clearly C.", VALID) == "C"

    def test_meandering_intro_then_letter(self):
        assert _extract_letter(
            "Looking at the data, it appears that the right choice is D.",
            VALID,
        ) == "D"

    def test_in_my_opinion_then_letter(self):
        assert _extract_letter("In my opinion, A is correct.", VALID) == "A"


class TestTieredFallback:
    def test_bare_letter_only(self):
        assert _extract_letter("B", VALID) == "B"

    def test_letter_with_period(self):
        assert _extract_letter("B.", VALID) == "B"

    def test_letter_with_paren(self):
        assert _extract_letter("(B)", VALID) == "B"

    def test_answer_colon_letter(self):
        assert _extract_letter("Answer: A", VALID) == "A"

    def test_answer_dash_letter(self):
        assert _extract_letter("Answer - C", VALID) == "C"

    def test_choice_phrase(self):
        assert _extract_letter("My choice is option (D).", VALID) == "D"


class TestValidChoiceFiltering:
    def test_letter_not_in_set_rejected(self):
        # The model wrote "Z"; not in valid set so we fall through.
        assert _extract_letter("Z", VALID) is None

    def test_only_in_set_letter_returned(self):
        # "First, the data shows X. Therefore, B."
        assert _extract_letter(
            "First, the data shows X. Therefore, B.", VALID,
        ) == "B"

    def test_no_letters_at_all(self):
        assert _extract_letter("the answer is unclear from the data", VALID) is None


class TestDefensive:
    def test_empty_string(self):
        assert _extract_letter("", VALID) is None

    def test_none_safe(self):
        assert _extract_letter(None, VALID) is None  # type: ignore[arg-type]

    def test_missing_valid_choices_still_finds_tag(self):
        assert _extract_letter("<answer>B</answer>", None) == "B"

    def test_multi_letter_tag_rejected(self):
        # Tag must contain a single letter.
        assert _extract_letter("<answer>BB</answer>", VALID) is None
