"""Directed Acyclic Graph for workflow management."""

import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional
import asyncio
import logging
import traceback

from ..utils.logging import get_logger
from .executors import BaseExecutor, LocalExecutor, HPCExecutor, KubernetesExecutor

logger = get_logger(__name__)

class WorkflowDAG:
    """Manages workflow as a Directed Acyclic Graph."""
    
    def __init__(self, executor_type: str = "local"):
        """Initialize workflow DAG.
        
        Args:
            executor_type: Type of executor to use ("local", "hpc", or "kubernetes")
        """
        self.graph = nx.DiGraph()
        
        # Initialize appropriate executor
        if executor_type == "kubernetes":
            self.executor = KubernetesExecutor()
        elif executor_type == "hpc":
            self.executor = HPCExecutor()
        else:
            self.executor = LocalExecutor()
            
        logger.info(f"Initialized WorkflowDAG with {executor_type} executor")
    
    def add_step(self, step: Dict[str, Any], dependencies: List[str] = None):
        """Add a step to the workflow graph."""
        self.graph.add_node(step["name"], step=step)
        
        if dependencies:
            for dep in dependencies:
                if dep not in self.graph:
                    raise ValueError(f"Dependency {dep} not found in graph")
                self.graph.add_edge(dep, step["name"])
                
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Dependencies would create a cycle in the graph")

    # ── Plan-patch surgery (LLM-driven pipeline-level recovery) ──────
    #
    # The reviewer of FlowAgent's recovery benchmark observed that the
    # original DAG was effectively immutable post-build: when a node failed,
    # only ``command`` on that node could be patched. That makes
    # *pipeline-level* fixes — "insert a ``samtools sort`` before this
    # ``samtools index`` step", or "swap two adjacent steps that are out of
    # order" — impossible by construction. ``apply_plan_patch`` lifts that
    # restriction so the LLM can return a structured pipeline edit when a
    # single-command rewrite cannot fix the failure.
    #
    # The supported actions are deliberately small and structural so the
    # operation can be audited:
    #
    #   * ``insert_before {target, new_step, deps?}``
    #   * ``insert_after  {target, new_step, deps?}``
    #   * ``replace_step  {target, new_step, deps?}``
    #   * ``remove_step   {target}``
    #
    # All actions preserve acyclicity and dependency-resolution; an
    # operation that would break either is rejected and rolled back.

    _ALLOWED_PATCH_ACTIONS = (
        "insert_before", "insert_after", "replace_step", "remove_step",
    )

    def apply_plan_patch(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a structured plan-patch to ``self.graph`` in-place.

        Returns ``{"status": "applied", "summary": "..."}`` on success, or
        ``{"status": "rejected", "reason": "..."}`` when the patch is
        malformed or would break the DAG. On rejection the graph is
        guaranteed to be untouched.
        """
        if not isinstance(patch, dict):
            return {"status": "rejected", "reason": "patch is not a dict"}
        action = patch.get("action")
        if action not in self._ALLOWED_PATCH_ACTIONS:
            return {
                "status": "rejected",
                "reason": f"unknown action: {action!r}",
            }
        target = patch.get("target")
        if not isinstance(target, str) or target not in self.graph:
            return {
                "status": "rejected",
                "reason": f"target step {target!r} not in DAG",
            }

        # Snapshot for rollback.
        snap_nodes = list(self.graph.nodes(data=True))
        snap_edges = list(self.graph.edges())

        try:
            if action == "remove_step":
                # Cannot remove a node that has dependents — they would
                # be orphaned. The LLM must propose a replace_step or a
                # restructure instead.
                successors = list(self.graph.successors(target))
                if successors:
                    return {
                        "status": "rejected",
                        "reason": (
                            f"cannot remove {target!r}; "
                            f"{len(successors)} downstream step(s) depend on it"
                        ),
                    }
                self.graph.remove_node(target)
                summary = f"removed step {target!r}"

            elif action == "replace_step":
                new_step = patch.get("new_step")
                if not isinstance(new_step, dict) or not new_step.get("name"):
                    return {
                        "status": "rejected",
                        "reason": "replace_step needs new_step with a name",
                    }
                if new_step["name"] != target and new_step["name"] in self.graph:
                    return {
                        "status": "rejected",
                        "reason": (
                            f"new_step name {new_step['name']!r} already in DAG"
                        ),
                    }
                preds = list(self.graph.predecessors(target))
                succs = list(self.graph.successors(target))
                self.graph.remove_node(target)
                self.graph.add_node(new_step["name"], step=new_step)
                for p in preds:
                    self.graph.add_edge(p, new_step["name"])
                for s in succs:
                    self.graph.add_edge(new_step["name"], s)
                summary = (
                    f"replaced step {target!r} with {new_step['name']!r}"
                )

            elif action in ("insert_before", "insert_after"):
                new_step = patch.get("new_step")
                if not isinstance(new_step, dict) or not new_step.get("name"):
                    return {
                        "status": "rejected",
                        "reason": f"{action} needs new_step with a name",
                    }
                if new_step["name"] in self.graph:
                    return {
                        "status": "rejected",
                        "reason": (
                            f"new_step name {new_step['name']!r} already in DAG"
                        ),
                    }
                extra_deps = patch.get("deps") or []
                if not isinstance(extra_deps, list):
                    return {
                        "status": "rejected",
                        "reason": "deps must be a list of step names",
                    }
                for d in extra_deps:
                    if d not in self.graph:
                        return {
                            "status": "rejected",
                            "reason": f"dependency {d!r} not in DAG",
                        }
                self.graph.add_node(new_step["name"], step=new_step)
                if action == "insert_before":
                    # Re-route every edge ``pred → target`` through the new
                    # step: ``pred → new → target``.
                    for pred in list(self.graph.predecessors(target)):
                        self.graph.remove_edge(pred, target)
                        self.graph.add_edge(pred, new_step["name"])
                    self.graph.add_edge(new_step["name"], target)
                else:  # insert_after
                    # Re-route every edge ``target → succ`` through new:
                    # ``target → new → succ``.
                    for succ in list(self.graph.successors(target)):
                        self.graph.remove_edge(target, succ)
                        self.graph.add_edge(new_step["name"], succ)
                    self.graph.add_edge(target, new_step["name"])
                for d in extra_deps:
                    self.graph.add_edge(d, new_step["name"])
                summary = (
                    f"{action} {new_step['name']!r} around {target!r}"
                )

            # Validate the resulting graph.
            if not nx.is_directed_acyclic_graph(self.graph):
                raise ValueError("plan-patch would introduce a cycle")
        except Exception as exc:
            # Roll back to the snapshot on any failure so a partially
            # applied patch never leaks into the running plan.
            self.graph.clear()
            self.graph.add_nodes_from(snap_nodes)
            self.graph.add_edges_from(snap_edges)
            return {"status": "rejected", "reason": str(exc)}

        return {"status": "applied", "summary": summary}
            
    def visualize(self, output_file: Path) -> Path:
        """Generate a visualization of the workflow DAG.
        
        Args:
            output_file: Path to save the visualization file
            
        Returns:
            Path to the generated visualization file or None if visualization fails
        """
        if output_file is None:
            logger.warning("No output file specified for workflow DAG visualization")
            return None
            
        try:
            if not self.graph.nodes:
                logger.warning("No nodes in workflow graph to visualize")
                return None
                
            # Create parent directory if it doesn't exist
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Clear any existing plot
            plt.clf()
            
            # Set up the plot with title
            plt.figure(figsize=(12, 8))
            plt.title("Workflow DAG", pad=20, size=16)
            
            # Use hierarchical layout for better workflow visualization
            pos = nx.spring_layout(self.graph, k=2, iterations=50)
            
            # Draw nodes with status-based colors
            node_colors = []
            completed_nodes = []
            failed_nodes = []
            pending_nodes = []
            start_nodes = []
            
            for node in self.graph.nodes:
                try:
                    # Get the step, handle case where key doesn't exist
                    step = self.graph.nodes[node].get("step")
                    if step is None:
                        logger.warning(f"Node {node} has no step data, skipping status check")
                        color = "#D3D3D3"  # Light gray for unknown status
                        pending_nodes.append(node)
                        node_colors.append(color)
                        continue
                    
                    # Check if step is a dictionary or an object
                    if isinstance(step, dict):
                        status = step.get("status", "pending")
                    else:
                        # Handle case where step is an object (like WorkflowStep)
                        status = getattr(step, "status", "pending")
                    
                    # Check if this is a starting node (no incoming edges)
                    is_start_node = self.graph.in_degree(node) == 0
                    
                    if is_start_node:
                        color = "#FFD700"  # Gold
                        start_nodes.append(node)
                    elif status == "completed":
                        color = "#90EE90"  # Light green
                        completed_nodes.append(node)
                    elif status in ["failed", "error"]:
                        color = "#FFA07A"  # Light salmon
                        failed_nodes.append(node)
                    else:
                        color = "#ADD8E6"  # Light blue
                        pending_nodes.append(node)
                    
                    node_colors.append(color)
                except Exception as e:
                    logger.warning(f"Error processing node {node}: {str(e)}")
                    color = "#D3D3D3"  # Light gray for unknown status
                    pending_nodes.append(node)
                    node_colors.append(color)
            
            # Draw nodes
            nx.draw_networkx_nodes(self.graph, pos,
                                 node_color=node_colors,
                                 node_size=2000,
                                 alpha=0.9)
            
            # Draw edges with better arrows
            nx.draw_networkx_edges(self.graph, pos,
                                 edge_color='gray',
                                 arrows=True,
                                 arrowsize=20,
                                 arrowstyle='->',
                                 width=1.5)
            
            # Add labels with better font
            nx.draw_networkx_labels(self.graph, pos,
                                  font_size=10,
                                  font_weight='bold')
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor='#FFD700', markersize=15, label='Start'),
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='#90EE90', markersize=15, label='Completed'),
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='#FFA07A', markersize=15, label='Failed'),
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='#ADD8E6', markersize=15, label='Pending')
            ]
            plt.legend(handles=legend_elements, loc='upper left', 
                      bbox_to_anchor=(1.05, 1), fontsize=10)
            
            # Add margin around the plot
            plt.margins(0.2)
            
            # Save the plot
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            plt.close()
            
            logger.info(f"Saved workflow visualization to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to visualize workflow: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _get_execution_batches(self) -> List[List[str]]:
        """Group steps into batches that can run in parallel.
        
        Each batch contains steps whose dependencies are all in earlier batches.
        """
        batches = []
        remaining = set(self.graph.nodes())
        completed = set()

        while remaining:
            batch = [
                node for node in remaining
                if all(pred in completed for pred in self.graph.predecessors(node))
            ]
            if not batch:
                raise ValueError("Cycle detected in workflow DAG")
            batches.append(batch)
            completed.update(batch)
            remaining -= set(batch)

        return batches

    async def execute_parallel(
        self,
        execute_fn: Optional[Callable] = None,
        recovery_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Execute workflow steps in parallel respecting dependencies.

        Steps within the same topological level run concurrently via
        asyncio.gather; levels are processed sequentially.

        Args:
            execute_fn: Optional function to execute steps. If not provided,
                       uses the configured executor.
            recovery_fn: Optional async callback ``(step, result) -> result|None``
                        called when a step fails.  If it returns a successful
                        result the step is treated as recovered and execution
                        continues.
        """
        try:
            execute = execute_fn or self.executor.execute_step
            jobs = {}
            batches = self._get_execution_batches()

            for batch in batches:
                async def _run_step(step_name: str) -> tuple:
                    step = self.graph.nodes[step_name]["step"]
                    step["dependencies"] = [
                        jobs[dep]["job_id"]
                        for dep in self.graph.predecessors(step_name)
                        if dep in jobs and "job_id" in jobs[dep]
                    ]
                    logger.info(f"Executing step: {step_name}")
                    result = await execute(step)
                    return step_name, result

                results_list = await asyncio.gather(
                    *[_run_step(name) for name in batch],
                    return_exceptions=True,
                )

                for item in results_list:
                    if isinstance(item, Exception):
                        raise item
                    step_name, result = item
                    jobs[step_name] = result
                    self.graph.nodes[step_name]["step"]["status"] = result.get("status", "pending")

                    # Persist a command-hash sidecar after success so
                    # smart-resume can detect when a planner-side command
                    # change requires re-execution on the next run.
                    # See ``write_step_state`` in
                    # ``flowagent/core/smart_resume.py``.
                    if result.get("status") == "completed":
                        try:
                            from flowagent.core.smart_resume import write_step_state
                            step_data = self.graph.nodes[step_name]["step"]
                            write_step_state(
                                step_name,
                                step_data.get("command", ""),
                                step_data.get("outputs", []),
                            )
                        except Exception as exc:  # pragma: no cover - defensive
                            logger.warning(
                                "Could not persist smart-resume state for "
                                "step %s: %s", step_name, exc,
                            )

                    if result.get("status") == "failed":
                        error_msg = result.get("stderr", "")
                        cmd = self.graph.nodes[step_name]["step"].get("command", "")
                        logger.error(f"Step {step_name} failed:\nCommand: {cmd}\nError: {error_msg}")

                        # Attempt LLM-driven recovery if a callback was provided.
                        # Only a re-executed step that returns status=="completed"
                        # counts as recovery. A "rejected" verdict (the LLM
                        # declined to propose a fix) must NOT be treated as
                        # success, or downstream dependent steps fire against
                        # missing outputs.
                        recovered = False
                        if recovery_fn is not None:
                            step_data = self.graph.nodes[step_name]["step"]
                            try:
                                recovery_result = await recovery_fn(step_data, result)
                                if recovery_result and recovery_result.get("status") == "completed":
                                    logger.info(f"Step {step_name} recovered successfully via LLM")
                                    jobs[step_name] = recovery_result
                                    self.graph.nodes[step_name]["step"]["status"] = "completed"
                                    # Update the command in the graph so downstream steps see the fix
                                    if recovery_result.get("fixed_command"):
                                        self.graph.nodes[step_name]["step"]["command"] = recovery_result["fixed_command"]
                                    recovered = True
                                    # Persist sidecar with the FIXED command, not
                                    # the broken original — otherwise a future
                                    # resume would see the broken command in the
                                    # plan and consider the step stale.
                                    try:
                                        from flowagent.core.smart_resume import write_step_state
                                        step_data = self.graph.nodes[step_name]["step"]
                                        write_step_state(
                                            step_name,
                                            step_data.get("command", ""),
                                            step_data.get("outputs", []),
                                        )
                                    except Exception as exc:  # pragma: no cover - defensive
                                        logger.warning(
                                            "Could not persist smart-resume state "
                                            "for recovered step %s: %s", step_name, exc,
                                        )
                                elif recovery_result and recovery_result.get("plan_patch"):
                                    # Pipeline-level fix: the LLM determined
                                    # the failure cannot be repaired by editing
                                    # the failing node's command alone (e.g. an
                                    # ``insert_before samtools sort`` is needed
                                    # ahead of ``samtools index``). Apply the
                                    # patch to the DAG and run the new/updated
                                    # node(s) before treating the original
                                    # failure as recovered.
                                    patch = recovery_result["plan_patch"]
                                    apply_res = self.apply_plan_patch(patch)
                                    if apply_res.get("status") == "applied":
                                        # Execute any newly-introduced node
                                        # (insert_before/after/replace) that
                                        # the patch targets so the failed
                                        # original step has its prerequisites
                                        # in place. We only run the patch's
                                        # ``new_step`` here; the original
                                        # failed step is re-run on the next
                                        # iteration of the outer batch loop
                                        # (which will pick up its updated
                                        # graph state).
                                        new_step = patch.get("new_step")
                                        ran_new = False
                                        if isinstance(new_step, dict) and new_step.get("name") in self.graph:
                                            try:
                                                logger.info(
                                                    "Executing patch-introduced step %s",
                                                    new_step["name"],
                                                )
                                                new_result = await execute(
                                                    self.graph.nodes[new_step["name"]]["step"]
                                                )
                                                jobs[new_step["name"]] = new_result
                                                self.graph.nodes[new_step["name"]]["step"]["status"] = (
                                                    new_result.get("status", "pending")
                                                )
                                                ran_new = new_result.get("status") == "completed"
                                            except Exception as patch_exec_err:
                                                logger.warning(
                                                    "Failed to execute patch-introduced step "
                                                    "%s: %s", new_step.get("name"), patch_exec_err,
                                                )
                                        # Re-run the original step now that the
                                        # patch has presumably set up what was
                                        # missing.
                                        if ran_new or patch.get("action") == "remove_step":
                                            try:
                                                rerun_step = self.graph.nodes[step_name]["step"] \
                                                    if step_name in self.graph else None
                                                if rerun_step is not None:
                                                    rerun_result = await execute(rerun_step)
                                                    jobs[step_name] = rerun_result
                                                    self.graph.nodes[step_name]["step"]["status"] = (
                                                        rerun_result.get("status", "pending")
                                                    )
                                                    if rerun_result.get("status") == "completed":
                                                        recovered = True
                                                        logger.info(
                                                            "Step %s recovered via DAG patch (%s)",
                                                            step_name, apply_res.get("summary"),
                                                        )
                                            except Exception as rerun_err:
                                                logger.warning(
                                                    "Re-run after DAG patch failed for %s: %s",
                                                    step_name, rerun_err,
                                                )
                                    else:
                                        logger.warning(
                                            "Plan-patch rejected for step %s: %s",
                                            step_name, apply_res.get("reason"),
                                        )
                                elif recovery_result and recovery_result.get("status") == "rejected":
                                    logger.error(
                                        "Step %s could not be recovered: %s",
                                        step_name,
                                        recovery_result.get("rejection_reason", "LLM declined to fix"),
                                    )
                            except Exception as rec_err:
                                logger.warning(f"Recovery callback failed for {step_name}: {rec_err}")

                        if not recovered:
                            if "command not found" in error_msg:
                                logger.error(f"Tool '{cmd.split()[0]}' not found. Please ensure it is installed and in your PATH")
                            elif "permission denied" in error_msg.lower():
                                logger.error("Permission denied. Check file/directory permissions")
                            elif "no such file" in error_msg.lower():
                                logger.error("Required input file not found. Check file paths and names")
                            raise Exception(f"Step {step_name} failed: {error_msg}")

            results = await self.executor.wait_for_completion(jobs)
            for step_name, result in results.items():
                self.graph.nodes[step_name]["step"]["status"] = result.get("status", "completed")

            return {
                "status": "success",
                "results": results
            }
            
        except Exception as e:
            # Mark remaining steps as failed and log detailed error
            failed_step = None
            for step_name in self.graph.nodes:
                status = self.graph.nodes[step_name]["step"].get("status", "")
                if status == "failed":
                    failed_step = step_name
                    break
                elif status != "completed":
                    self.graph.nodes[step_name]["step"]["status"] = "cancelled"
            
            if failed_step:
                step = self.graph.nodes[failed_step]["step"]
                logger.error(f"Workflow failed at step '{failed_step}':")
                logger.error(f"Command: {step.get('command', 'N/A')}")
                logger.error(f"Error: {str(e)}")
                logger.error("Dependencies:")
                for dep in self.graph.predecessors(failed_step):
                    dep_status = self.graph.nodes[dep]["step"].get("status", "unknown")
                    logger.error(f"  - {dep}: {dep_status}")
            
            return {
                "status": "failed",
                "error": str(e),
                "failed_step": failed_step
            }
