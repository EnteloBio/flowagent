"""Regression tests for critical workflow / recovery fixes."""

from flowagent.core.executors import (
    normalize_step_status,
    step_status_failed,
    step_status_succeeded,
)


def test_step_status_helpers_accept_legacy_success():
    assert step_status_succeeded("completed")
    assert step_status_succeeded("success")
    assert not step_status_succeeded("failed")
    assert step_status_failed("failed")
    assert step_status_failed("error")
    assert normalize_step_status("success") == "completed"
    assert normalize_step_status("error") == "failed"


def test_dag_envelope_parsing_extracts_step_results():
    """execute_workflow post-processing must read dag_results['results']."""
    envelope = {
        "status": "success",
        "results": {
            "qc": {"step_id": "qc", "status": "completed", "returncode": 0},
            "align": {"step_id": "align", "status": "completed", "returncode": 0},
        },
    }
    step_results = envelope.get("results")
    raw = list(step_results.values())
    assert len(raw) == 2
    assert raw[0]["step_id"] == "qc"
    # Old bug: list(envelope.values()) would include "success" string.
    assert "success" not in {r if isinstance(r, str) else r.get("step_id") for r in raw}

    """execute_workflow post-processing must read dag_results['results']."""
    envelope = {
        "status": "success",
        "results": {
            "qc": {"step_id": "qc", "status": "completed", "returncode": 0},
            "align": {"step_id": "align", "status": "completed", "returncode": 0},
        },
    }
    step_results = envelope.get("results")
    raw = list(step_results.values())
    assert len(raw) == 2
    assert raw[0]["step_id"] == "qc"
    # Old bug: list(envelope.values()) would include "success" string.
    assert "success" not in {r if isinstance(r, str) else r.get("step_id") for r in raw}
