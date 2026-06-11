"""Unit tests for the calibrated judge prompt + parser.

The reviewer of FlowAgent's open-ended interpretation grading
observed that the judge had no anchored bands and no pass mark, and
that the parser only carried score+justification. The fix:

  - Adds five anchored score bands to ``_SYSTEM_JUDGE``.
  - States the pass mark (≥60) explicitly in ``_judge_prompt``.
  - Asks for a structured JSON schema with rubric ``hits``,
    ``misses``, ``fabrications``, ``grounding_quote`` and
    ``justification``.
  - The parser ``_parse_judge_json`` returns a dict with all those
    fields; missing ones collapse to safe defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(BENCH_DIR.parent))

from bench_interpretation import (  # noqa: E402
    _SYSTEM_JUDGE, _PASS_MARK, _judge_prompt, _parse_judge_json,
)


_QUESTION = {
    "question": "What is the dominant chromosome in the supplied peaks?",
    "rubric":   "Names the most frequent chromosome and cites a count.",
    "reference_answer": "chr1, with ~3500 peaks.",
}


class TestSystemJudgeAnchors:
    def test_anchors_listed(self):
        for band in ["0-20", "21-40", "41-59", "60-79", "80-100"]:
            assert band in _SYSTEM_JUDGE, (
                f"score band {band!r} missing from judge system prompt"
            )

    def test_pass_mark_named(self):
        assert "60" in _SYSTEM_JUDGE
        assert "pass mark" in _SYSTEM_JUDGE.lower()


class TestJudgePromptContents:
    def test_includes_question_rubric_reference(self):
        prompt = _judge_prompt(_QUESTION, "candidate text")
        assert _QUESTION["question"] in prompt
        assert _QUESTION["rubric"] in prompt
        assert _QUESTION["reference_answer"] in prompt
        assert "candidate text" in prompt

    def test_states_pass_mark(self):
        prompt = _judge_prompt(_QUESTION, "candidate text")
        assert "60" in prompt
        # The PASS MARK statement must come *after* the system anchors
        # so the LLM cannot miss it.
        assert "PASS MARK" in prompt

    def test_requests_structured_json_keys(self):
        prompt = _judge_prompt(_QUESTION, "candidate text")
        for key in ["score", "hits", "misses", "fabrications",
                    "grounding_quote", "justification"]:
            assert f'"{key}"' in prompt, f"key {key!r} not in prompt"


class TestParseJudgeJson:
    def test_full_payload_round_trips(self):
        reply = (
            'Some preamble.\n'
            '{"score": 72, "hits": ["named chr1", "cited count"], '
            '"misses": ["did not mention chr2"], '
            '"fabrications": [], '
            '"grounding_quote": "chr1 has ~3500 peaks", '
            '"justification": "covers most rubric items"}'
            '\nTrailing prose.'
        )
        out = _parse_judge_json(reply)
        assert out["score"] == 72.0
        assert out["hits"] == ["named chr1", "cited count"]
        assert out["misses"] == ["did not mention chr2"]
        assert out["fabrications"] == []
        assert out["grounding_quote"] == "chr1 has ~3500 peaks"
        assert "covers" in out["justification"]

    def test_score_clipped_to_range(self):
        out = _parse_judge_json('{"score": 250, "justification": "x"}')
        assert out["score"] == 100.0
        out2 = _parse_judge_json('{"score": -10, "justification": "x"}')
        assert out2["score"] == 0.0

    def test_missing_score_returns_none(self):
        out = _parse_judge_json('{"justification": "x"}')
        assert out["score"] is None

    def test_missing_lists_default_empty(self):
        out = _parse_judge_json('{"score": 50}')
        assert out["hits"] == []
        assert out["misses"] == []
        assert out["fabrications"] == []
        assert out["grounding_quote"] == ""

    def test_string_for_list_field_coerced(self):
        out = _parse_judge_json('{"score": 60, "hits": "named chr1"}')
        assert out["hits"] == ["named chr1"]

    def test_invalid_json_returns_safe_default(self):
        out = _parse_judge_json("the model wandered off and wrote prose")
        assert out["score"] is None
        assert out["hits"] == []
        assert "wandered" in out["justification"]

    def test_empty_reply_safe(self):
        out = _parse_judge_json("")
        assert out["score"] is None
        assert out["justification"] == ""


class TestPassMarkConstant:
    def test_pass_mark_is_60(self):
        assert _PASS_MARK == 60.0
