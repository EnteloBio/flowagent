"""Unit tests for ``WorkflowDAG.apply_plan_patch``.

The reviewer of FlowAgent's recovery benchmark observed that the DAG
was effectively immutable post-build — only ``command`` on a failed
node could be patched, never the pipeline structure. The fix added
``apply_plan_patch`` for ``insert_before``, ``insert_after``,
``replace_step``, ``remove_step``. These tests pin the contract:

  - structural changes apply when valid,
  - acyclicity is preserved (cycles → reject + rollback),
  - dangling references are rejected,
  - removing a step with downstream dependents is rejected (would
    orphan them),
  - the rollback path leaves the graph untouched on failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from flowagent.core.workflow_dag import WorkflowDAG  # noqa: E402


def _build_linear_pipeline() -> WorkflowDAG:
    """Build a → b → c linear pipeline."""
    dag = WorkflowDAG(executor_type="local")
    dag.add_step({"name": "a", "command": "echo a"})
    dag.add_step({"name": "b", "command": "echo b"}, dependencies=["a"])
    dag.add_step({"name": "c", "command": "echo c"}, dependencies=["b"])
    return dag


class TestInsertBefore:
    def test_inserts_node_and_reroutes_edges(self):
        dag = _build_linear_pipeline()
        result = dag.apply_plan_patch({
            "action":   "insert_before",
            "target":   "b",
            "new_step": {"name": "a2", "command": "echo a2"},
        })
        assert result["status"] == "applied"
        assert "a2" in dag.graph
        # ``a`` should now point at ``a2`` and ``a2`` should point at ``b``.
        assert dag.graph.has_edge("a", "a2")
        assert dag.graph.has_edge("a2", "b")
        # Old ``a → b`` edge must be gone.
        assert not dag.graph.has_edge("a", "b")
        assert nx.is_directed_acyclic_graph(dag.graph)

    def test_first_node_insert_before_creates_root_predecessor(self):
        dag = _build_linear_pipeline()
        result = dag.apply_plan_patch({
            "action":   "insert_before",
            "target":   "a",
            "new_step": {"name": "a0", "command": "echo a0"},
        })
        assert result["status"] == "applied"
        assert dag.graph.has_edge("a0", "a")


class TestInsertAfter:
    def test_inserts_node_after_target(self):
        dag = _build_linear_pipeline()
        result = dag.apply_plan_patch({
            "action":   "insert_after",
            "target":   "b",
            "new_step": {"name": "b2", "command": "echo b2"},
        })
        assert result["status"] == "applied"
        assert dag.graph.has_edge("b", "b2")
        assert dag.graph.has_edge("b2", "c")
        assert not dag.graph.has_edge("b", "c")
        assert nx.is_directed_acyclic_graph(dag.graph)


class TestReplaceStep:
    def test_replaces_node_preserving_neighbours(self):
        dag = _build_linear_pipeline()
        result = dag.apply_plan_patch({
            "action":   "replace_step",
            "target":   "b",
            "new_step": {"name": "b_fixed", "command": "echo b_fixed"},
        })
        assert result["status"] == "applied"
        assert "b" not in dag.graph
        assert "b_fixed" in dag.graph
        assert dag.graph.has_edge("a", "b_fixed")
        assert dag.graph.has_edge("b_fixed", "c")

    def test_replace_with_existing_name_rejects(self):
        dag = _build_linear_pipeline()
        result = dag.apply_plan_patch({
            "action":   "replace_step",
            "target":   "b",
            "new_step": {"name": "c", "command": "echo c"},
        })
        assert result["status"] == "rejected"
        # original DAG untouched
        assert "b" in dag.graph
        assert "c" in dag.graph


class TestRemoveStep:
    def test_remove_leaf(self):
        dag = _build_linear_pipeline()
        result = dag.apply_plan_patch({"action": "remove_step",
                                       "target": "c"})
        assert result["status"] == "applied"
        assert "c" not in dag.graph

    def test_remove_node_with_dependents_rejects(self):
        dag = _build_linear_pipeline()
        result = dag.apply_plan_patch({"action": "remove_step",
                                       "target": "b"})
        assert result["status"] == "rejected"
        # Both b and c (its dependent) untouched.
        assert "b" in dag.graph
        assert "c" in dag.graph


class TestValidationAndRollback:
    def test_unknown_action_rejects(self):
        dag = _build_linear_pipeline()
        result = dag.apply_plan_patch({"action": "obliterate",
                                       "target": "a"})
        assert result["status"] == "rejected"
        assert set(dag.graph.nodes) == {"a", "b", "c"}

    def test_missing_target_rejects(self):
        dag = _build_linear_pipeline()
        result = dag.apply_plan_patch({"action": "insert_before",
                                       "target": "ghost",
                                       "new_step": {"name": "x",
                                                    "command": "echo x"}})
        assert result["status"] == "rejected"
        assert "x" not in dag.graph

    def test_dep_not_in_graph_rejects(self):
        dag = _build_linear_pipeline()
        result = dag.apply_plan_patch({
            "action":   "insert_before",
            "target":   "b",
            "new_step": {"name": "a2", "command": "echo a2"},
            "deps":     ["nonexistent"],
        })
        assert result["status"] == "rejected"
        # The node was tentatively added then rolled back on rejection.
        assert "a2" not in dag.graph

    def test_extra_dep_creating_cycle_rejected_with_rollback(self):
        dag = _build_linear_pipeline()
        # Forcing ``a2`` to depend on ``c`` as an extra-dep while we
        # also rewire ``a → a2 → b`` makes the graph non-DAG
        # (c → a2 → b, but c is a successor of b → cycle).
        result = dag.apply_plan_patch({
            "action":   "insert_before",
            "target":   "b",
            "new_step": {"name": "a2", "command": "echo a2"},
            "deps":     ["c"],
        })
        assert result["status"] == "rejected"
        assert "a2" not in dag.graph
        assert nx.is_directed_acyclic_graph(dag.graph)
        # Original edges intact after rollback.
        assert dag.graph.has_edge("a", "b")
        assert dag.graph.has_edge("b", "c")

    def test_non_dict_patch_rejects(self):
        dag = _build_linear_pipeline()
        result = dag.apply_plan_patch("not a dict")  # type: ignore[arg-type]
        assert result["status"] == "rejected"
