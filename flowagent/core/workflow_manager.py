"""Workflow manager for coordinating LLM-based workflow execution."""

from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
import asyncio
import hashlib
import networkx as nx
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from flowagent.config.settings import Settings

from ..utils.logging import get_logger
from ..utils import file_utils
from ..utils.dependency_manager import DependencyManager
from .llm import LLMInterface
from .executor import Executor
from .executor_factory import ExecutorFactory
from .agent_types import WorkflowStep, Workflow
from .workflow_dag import WorkflowDAG
from .smart_resume import detect_completed_steps, filter_workflow_steps
from ..agents.agentic.analysis_system import AgenticAnalysisSystem

logger = get_logger(__name__)

class WorkflowManager:
    """Manages workflow execution and coordination."""
    
    def __init__(self, executor_type: str = "local"):
        """Initialize the workflow manager.
        
        Args:
            executor_type: Type of executor to use (local, slurm, etc.)
        """
        self.logger = get_logger(__name__)
        self.llm = LLMInterface()
        self.dependency_manager = DependencyManager()
        self.executor_type = executor_type

        # Get settings
        self.settings = Settings()

        # Special handling for Kubernetes executor
        if self.executor_type == "kubernetes" and not self.settings.KUBERNETES_ENABLED:
            self.logger.warning("Kubernetes executor requested but not enabled in settings. Defaulting to 'local'")
            self.executor_type = "local"

        # Factory executor for advanced backends (CGAT, HPC, Kubernetes, Nextflow, Snakemake)
        self.executor = ExecutorFactory.create(self.executor_type)
        # Legacy Executor wraps subprocess-based step execution for local/slurm
        self._legacy_executor = Executor(executor_type if executor_type in ("local", "slurm") else "local")
        # Prefer factory executor when it supports execute_step
        self._step_executor = (
            self.executor if hasattr(self.executor, "execute_step") else self._legacy_executor
        )

        # Get initial working directory
        self.initial_cwd = os.getcwd()
        self.logger.info(f"Initial working directory: {self.initial_cwd}")
        self.logger.info(f"Using {self.executor_type} executor (via ExecutorFactory)")
            
        self.analysis_system = AgenticAnalysisSystem()

    @staticmethod
    def _file_checksum(filepath: str) -> str:
        """Compute SHA-256 checksum for a file (first 10 MB)."""
        h = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
                    if f.tell() > 10 * (1 << 20):
                        break
            return h.hexdigest()
        except Exception:
            return "unavailable"

    def _write_manifest(self, output_dir: str, prompt: Optional[str],
                        workflow_plan: Dict[str, Any],
                        results: List[Dict[str, Any]]):
        """Emit workflow_manifest.json for reproducibility."""
        settings = self.settings

        # Collect input file checksums from the plan
        input_checksums = {}
        for step in workflow_plan.get("steps", []):
            for output_pattern in step.get("outputs", []):
                import glob as _glob
                for f in _glob.glob(os.path.join(output_dir, output_pattern)):
                    if os.path.isfile(f):
                        input_checksums[os.path.relpath(f, output_dir)] = self._file_checksum(f)

        manifest = {
            "flowagent_version": "0.2.0",
            "prompt": prompt,
            "llm_provider": settings.LLM_PROVIDER,
            "llm_model": settings.LLM_MODEL,
            "executor_type": self.executor_type,
            "workflow_plan": workflow_plan,
            "step_results": [
                {"name": r.get("step_name", "?"), "status": r.get("status", "?")}
                for r in results
            ],
            "output_checksums": input_checksums,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        manifest_path = os.path.join(output_dir, "workflow_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        self.logger.info("Manifest written to %s", manifest_path)

    def _write_checkpoint(self, checkpoint_dir: str, workflow_plan: Dict[str, Any],
                          output_dir: str, completed_steps: List[str],
                          prompt: Optional[str] = None):
        """Persist checkpoint.json so that resume_workflow can reload state."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = {
            "workflow_plan": workflow_plan,
            "output_dir": output_dir,
            "completed_steps": completed_steps,
            "prompt": prompt,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        path = os.path.join(checkpoint_dir, "checkpoint.json")
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        self.logger.debug("Checkpoint written: %s (%d steps done)", path, len(completed_steps))

    # ── LLM error recovery ────────────────────────────────────

    @staticmethod
    def _extract_executables(command: str) -> List[str]:
        """Extract executable names from a shell command string."""
        SHELL_BUILTINS = frozenset({
            "for", "do", "done", "if", "then", "else", "fi", "while",
            "until", "case", "esac", "in", "export", "cd", "echo",
            "true", "false", "test", "[", "[[", "set", "unset",
            "local", "return", "break", "continue", "shift",
        })
        segments = re.split(r'\s*[|;&]+\s*', command)
        executables = []
        for seg in segments:
            tokens = seg.strip().split()
            if not tokens:
                continue
            first = tokens[0]
            # Skip builtins, variable assignments, sub-shell parens
            if first in SHELL_BUILTINS or "=" in first or first.startswith("("):
                continue
            # Strip leading path (e.g. /usr/bin/env)
            executables.append(os.path.basename(first))
        return list(dict.fromkeys(executables))  # dedupe, preserve order

    async def _attempt_error_recovery(
        self,
        step: Dict[str, Any],
        step_result: Dict[str, Any],
        output_dir: str,
        attempt: int = 1,
        max_attempts: int = 3,
    ) -> Optional[Dict[str, Any]]:
        """Feed a step failure to the LLM to get a fixed command, then retry.

        Returns the successful step result, or None if recovery is not possible.
        """
        if attempt > max_attempts:
            self.logger.error(
                "Max recovery attempts (%d) reached for step '%s'",
                max_attempts, step.get("name"),
            )
            return None

        # Short-circuit: if the failure is a Python SyntaxError inside
        # imported code, no shell-level edit can fix it. Skip the LLM
        # round-trip and return a structured rejection — this saves 3 API
        # calls and stops the LLM from hallucinating non-existent CLI flags.
        stderr_tail = (step_result.get("stderr") or "") + (step_result.get("stdout") or "")
        if (
            "SyntaxError" in stderr_tail
            and ("f-string" in stderr_tail or "cannot include a backslash" in stderr_tail
                 or "invalid syntax" in stderr_tail)
        ):
            self.logger.error(
                "Step '%s' failed with a Python SyntaxError in imported code — "
                "not recoverable via shell edits. Fix the source and re-run.",
                step.get("name"),
            )
            return {
                "status": "rejected",
                "recovery_attempt": attempt,
                "recovery_diagnosis": "Python SyntaxError in imported module — source-code bug, not a shell-level issue.",
                "rejection_reason": "No shell-level fix can resolve a Python SyntaxError.",
                "fixed_command": None,
                "original_command": step.get("command"),
                "step_name": step.get("name", "unknown"),
            }

        # Build tool-availability context
        executables = self._extract_executables(step.get("command", ""))
        tool_availability = {cmd: shutil.which(cmd) is not None for cmd in executables}

        error_context = {
            "step_name": step.get("name"),
            "step_description": step.get("description", ""),
            "original_command": step.get("command"),
            "exit_code": step_result.get("exit_code") or step_result.get("returncode"),
            "stderr": (step_result.get("stderr") or "")[:3000],
            "stdout": (step_result.get("stdout") or "")[:1500],
            "platform": "macOS" if sys.platform == "darwin" else "Linux",
            "tool_availability": tool_availability,
            "attempt": attempt,
        }

        prompt = (
            "A bioinformatics pipeline step failed during execution. "
            "Diagnose the error and return a fixed shell command.\n\n"
            f"Step name: {error_context['step_name']}\n"
            f"Step description: {error_context['step_description']}\n"
            f"Original command:\n  {error_context['original_command']}\n"
            f"Exit code: {error_context['exit_code']}\n"
            f"stderr:\n  {error_context['stderr']}\n"
            f"stdout (tail):\n  {error_context['stdout']}\n"
            f"Platform: {error_context['platform']}\n"
            f"Tool availability: {json.dumps(error_context['tool_availability'])}\n"
            f"Recovery attempt: {attempt}/{max_attempts}\n\n"
            "Common fixes by error class:\n"
            "- Exit code 127 (command not found): substitute an equivalent tool "
            "(e.g. curl -fSL -o <file> <url> instead of wget, pigz instead of gzip).\n"
            "- 'No such file or directory': add mkdir -p for missing directories.\n"
            "- Permission denied: check paths and permissions.\n"
            "- R/Bioconductor errors:\n"
            "  * tximport 'None of the transcripts ... present in the first column of tx2gene': "
            "add ignoreTxVersion=TRUE, ignoreAfterBar=TRUE to tximport(); also strip version "
            "suffixes from tx2gene TXNAME via sub('\\\\..*$','',...). Ensembl cDNA FASTA carries "
            "version suffixes, GTF transcript_id does not.\n"
            "  * DESeq2 'colData rownames do not match countData colnames': intersect sample "
            "names and subset both (txi$counts <- txi$counts[, keep]).\n"
            "  * 'could not find function \"X\"' / 'there is no package called X': add "
            "library(X) or install via BiocManager::install('X').\n"
            "  * rtracklayer 'duplicate row names': dedupe with unique() on the data.frame.\n"
            "- Python module errors (ModuleNotFoundError): install via pip in the right env, "
            "or rewrite using a stdlib equivalent.\n"
            "- When rewriting R one-liners, preserve the existing Rscript -e '...' shape and "
            "all input/output paths from the original command — only change the R logic.\n"
            "- In R regex within Rscript -e, prefer a character class over backslash escapes: "
            "use '[.].*$' instead of '\\\\..*$' to match a literal dot. Modern R (4.4+) errors "
            "on '\\.' as an unrecognised escape at parse time, and JSON payloads lose one "
            "backslash layer on the wire — character classes sidestep both.\n"
            "- In the JSON response, the ``fixed_command`` string MUST be valid JSON: every "
            "backslash must be escaped as \\\\, and every double-quote as \\\". If you're "
            "unsure about escaping, rewrite the regex with a character class instead.\n"
            "- CRITICAL: if the error is a bug inside imported source code (Python "
            "SyntaxError / ImportError from an imported module, NameError in library "
            "code, R parse error in an installed package, etc.), NO shell-level edit "
            "can fix it. DO NOT invent CLI flags (--template, --no-fstring, etc.) that "
            "aren't documented in the original command. Return null as the "
            "``fixed_command`` with a diagnosis explaining the source-code bug.\n\n"
            "Return ONLY a JSON object with this structure:\n"
            '{"diagnosis": "short explanation", '
            '"fixed_command": "the corrected shell command or null if unrecoverable", '
            '"explanation": "what you changed and why"}'
        )

        try:
            response = await self.llm._call_openai([
                {
                    "role": "system",
                    "content": (
                        "You are an expert at diagnosing and fixing bioinformatics "
                        "pipeline failures. Return valid JSON only, no markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ])

            cleaned = self.llm._clean_llm_response(response)
            fix = json.loads(cleaned)

            fixed_command = fix.get("fixed_command")
            diagnosis = fix.get("diagnosis", "")
            explanation = fix.get("explanation", "")

            if not fixed_command:
                self.logger.warning(
                    "LLM determined step '%s' is unrecoverable: %s",
                    step.get("name"), diagnosis,
                )
                # Return a structured refusal instead of ``None`` so callers
                # (notably the recovery benchmark) can record *why* the LLM
                # declined — on unrecoverable faults, the refusal text is
                # the primary datum. Preserves backward compatibility
                # because ``status`` is still non-success.
                return {
                    "status": "rejected",
                    "recovery_attempt": attempt,
                    "recovery_diagnosis": diagnosis,
                    "rejection_reason": explanation or (
                        "LLM determined the step is unrecoverable"
                    ),
                    "fixed_command": None,
                    "raw_response": response,
                    "original_command": step.get("command"),
                    "step_name": step.get("name", "unknown"),
                }

            self.logger.info(
                "LLM recovery attempt %d for '%s': %s",
                attempt, step.get("name"), explanation,
            )
            self.logger.info("Fixed command: %s", fixed_command)

            # Build a patched step and re-execute
            fixed_step = {**step, "command": fixed_command}
            new_result = await self._step_executor.execute_step(fixed_step)
            new_result["step_name"] = step.get("name", "unknown")
            new_result["recovery_attempt"] = attempt
            new_result["recovery_diagnosis"] = diagnosis
            new_result["original_command"] = step.get("command")
            new_result["fixed_command"] = fixed_command

            if new_result.get("status") in ("error", "failed"):
                # Recurse with the fixed step so subsequent attempts build on each fix
                return await self._attempt_error_recovery(
                    fixed_step, new_result, output_dir,
                    attempt=attempt + 1, max_attempts=max_attempts,
                )

            self.logger.info(
                "Step '%s' recovered successfully on attempt %d",
                step.get("name"), attempt,
            )
            return new_result

        except asyncio.TimeoutError:
            self.logger.warning(
                "LLM recovery call timed out for step '%s' (attempt %d)",
                step.get("name"), attempt,
            )
            return None
        except (json.JSONDecodeError, KeyError) as parse_err:
            self.logger.warning(
                "Failed to parse LLM recovery response (attempt %d): %s",
                attempt, parse_err,
            )
            return None
        except Exception as exc:
            self.logger.warning(
                "Error during recovery attempt %d for '%s': %s",
                attempt, step.get("name"), exc,
            )
            return None

    async def execute_workflow(self, prompt_or_workflow: Union[str, Workflow]) -> Dict[str, Any]:
        """Execute workflow from prompt or workflow object."""
        try:
            # Check if input is a workflow object or a prompt string
            if isinstance(prompt_or_workflow, Workflow):
                # Use the provided workflow object
                workflow = {
                    "name": prompt_or_workflow.name,
                    "description": prompt_or_workflow.description,
                    "steps": [vars(step) for step in prompt_or_workflow.steps]
                }
                workflow_name = workflow["name"]
                workflow_steps = workflow["steps"]
                prompt = None  # No prompt in this case
                
                # Create output directory if needed
                if prompt_or_workflow.output_dir:
                    output_dir = prompt_or_workflow.output_dir
                else:
                    output_dir = os.path.join(self.initial_cwd, "flowagent_output", workflow_name.replace(" ", "_"))
                
                os.makedirs(output_dir, exist_ok=True)
                
                # Save workflow to output directory
                workflow_file = os.path.join(output_dir, "workflow.json")
                with open(workflow_file, "w") as f:
                    json.dump(workflow, f, indent=2)
                
                self.logger.info(f"Saved workflow to {workflow_file}")
                
            else:
                # Generate workflow from prompt
                prompt = prompt_or_workflow
                workflow_data = await self.llm.generate_workflow_plan(prompt)
                
                # Extract workflow from response
                workflow_name = workflow_data.get("workflow_type", "Unnamed workflow")
                workflow_steps = workflow_data.get("steps", [])
                workflow = {"name": workflow_name, "steps": workflow_steps}
                
                self.logger.info(f"Generated workflow: {workflow_name} with {len(workflow_steps)} steps")
                
                # Create output directory
                output_dir = os.path.join(self.initial_cwd, "flowagent_output", workflow_name.replace(" ", "_"))
                os.makedirs(output_dir, exist_ok=True)
                
                # Save workflow to output directory
                workflow_file = os.path.join(output_dir, "workflow.json")
                with open(workflow_file, "w") as f:
                    json.dump(workflow, f, indent=2)
                
                self.logger.info(f"Saved workflow to {workflow_file}")
                
                # Check for GEO accession in prompt and add download steps if needed
                if prompt:
                    geo_accession = self._extract_geo_accession(prompt)
                    if geo_accession:
                        self.logger.info(f"Detected GEO accession: {geo_accession}")
                        workflow_steps = self._add_geo_download_steps(geo_accession, workflow_steps, output_dir)
                        
                        # Update workflow file with new steps
                        workflow["steps"] = workflow_steps
                        with open(workflow_file, "w") as f:
                            json.dump(workflow, f, indent=2)
            
            # Check and install dependencies
            self.logger.info("Checking dependencies...")
            
            # Create a workflow plan with dependencies for the dependency manager
            workflow_plan = {
                "name": workflow_name,
                "steps": workflow_steps
            }
            
            # Try to ensure all dependencies are installed
            all_installed, available_but_failed_install = self.dependency_manager.ensure_workflow_dependencies_sync(workflow_plan)
            
            if not all_installed:
                self.logger.warning("Not all dependencies could be installed")
                
                # Check dependencies to get the missing ones
                dependency_results = self.dependency_manager.check_dependencies(workflow_steps)
                missing_dependencies = dependency_results.get("missing", [])
                
                # Remove tools that are available but failed to install from missing dependencies
                missing_dependencies = [dep for dep in missing_dependencies if dep not in available_but_failed_install]
                
                if missing_dependencies:
                    self.logger.warning(f"Missing dependencies: {', '.join(missing_dependencies)}")
                    self.logger.warning("Some workflow steps may fail due to missing dependencies")
                    
                    # Check if critical tools are missing
                    critical_missing = False
                    for step in workflow_steps:
                        if step.get("critical", False):
                            step_tools = step.get("tools", [])
                            missing_critical_tools = [tool for tool in step_tools if tool in missing_dependencies]
                            if missing_critical_tools:
                                self.logger.error(f"Critical step {step.get('name')} requires missing tools: {', '.join(missing_critical_tools)}")
                                critical_missing = True
                    
                    if critical_missing:
                        self.logger.warning("Critical tools are missing, but attempting to execute workflow anyway")
                else:
                    self.logger.info("All dependencies are available or can be used despite installation failures")
            else:
                self.logger.info("All dependencies are available")
            
            # Execute workflow steps
            self.logger.info(f"Executing workflow: {workflow_name}")

            # Use DAG-parallel execution when steps declare dependencies
            has_deps = any(step.get("dependencies") for step in workflow_steps)
            dag_executed = False  # True only if the DAG path ran end-to-end
            if has_deps and len(workflow_steps) > 1:
                try:
                    dag = WorkflowDAG(executor_type=self.executor_type)
                    for step in workflow_steps:
                        dag.add_step(step, dependencies=step.get("dependencies", []))
                    self.logger.info("Using DAG-parallel execution (%d steps)", len(workflow_steps))
                    # Build a recovery callback for the DAG executor
                    async def _dag_recovery(step, result):
                        return await self._attempt_error_recovery(step, result, output_dir)

                    dag_results = await dag.execute_parallel(
                        self._step_executor.execute_step,
                        recovery_fn=_dag_recovery,
                    )
                    results = list(dag_results.values()) if isinstance(dag_results, dict) else dag_results
                    # Normalise results to list-of-dicts with step_name
                    for r in results:
                        if isinstance(r, dict) and "step_name" not in r:
                            r["step_name"] = r.get("step_id", r.get("name", "unknown"))
                    # Write final checkpoint
                    completed = [r.get("step_name", "") for r in results if r.get("status") not in ("error", "failed")]
                    ckpt_dir = getattr(prompt_or_workflow, "checkpoint_dir", None) or os.path.join(output_dir, ".checkpoint")
                    self._write_checkpoint(ckpt_dir, workflow_plan, output_dir, completed, prompt)
                    dag_executed = True
                except (ValueError, Exception) as dag_err:
                    self.logger.warning("DAG-parallel execution failed (%s), falling back to sequential", dag_err)
                    results = []
                    has_deps = False  # fall through to sequential

            if not has_deps or not results:
                results = []

            for i, step in enumerate(workflow_steps):
                if dag_executed:
                    break  # already executed via DAG — don't re-run in sequential
                step_name = step.get("name", f"Step {i+1}")
                self.logger.info(f"Executing step {i+1}/{len(workflow_steps)}: {step_name}")
                
                # Get dependency results to check for missing tools
                dependency_results = self.dependency_manager.check_dependencies([step])
                missing_dependencies = dependency_results.get("missing", [])
                
                # Remove tools that are available but failed to install from missing dependencies
                missing_dependencies = [dep for dep in missing_dependencies if dep not in available_but_failed_install]
                
                # Check if dependencies for this step are available
                step_tools = step.get("tools", [])
                missing_step_tools = [tool for tool in step_tools if tool in missing_dependencies]
                
                if missing_step_tools:
                    self.logger.warning(f"Step {step_name} requires missing tools: {', '.join(missing_step_tools)}")
                    
                    # If this is a critical step and tools are missing, we might want to skip it
                    if step.get("critical", False) and len(missing_step_tools) == len(step_tools):
                        self.logger.error(f"Critical step {step_name} is missing all required tools, skipping")
                        results.append({
                            "status": "skipped",
                            "step": step,
                            "reason": f"Missing required tools: {', '.join(missing_step_tools)}"
                        })
                        continue
                    
                    self.logger.warning(f"Attempting to execute step anyway...")
                
                # Execute step
                step_result = await self._step_executor.execute_step(step)
                
                # Add step name to the result for easier reference
                step_result["step_name"] = step_name
                
                results.append(step_result)

                # Write checkpoint after every step
                completed = [r["step_name"] for r in results if r.get("status") not in ("error", "failed")]
                ckpt_dir = getattr(prompt_or_workflow, "checkpoint_dir", None) or os.path.join(output_dir, ".checkpoint")
                self._write_checkpoint(ckpt_dir, workflow_plan, output_dir, completed, prompt)
                
                # Check if step failed (accept both legacy "error" and canonical "failed")
                step_status = step_result.get("status", "")
                if step_status in ("error", "failed"):
                    error_msg = step_result.get('error', '')
                    stderr = step_result.get('stderr', '')

                    self.logger.error(f"Step {step_name} failed: {error_msg}")
                    if stderr:
                        self.logger.error(f"STDERR: {stderr}")

                    # Attempt LLM-driven error recovery
                    self.logger.info(f"Attempting LLM error recovery for step '{step_name}'...")
                    recovery_result = await self._attempt_error_recovery(
                        step, step_result, output_dir,
                    )
                    # Positive allowlist — status must explicitly signal success.
                    # The helper returns status="rejected" for unrecoverable cases,
                    # which previously slipped through a negative check and got
                    # logged as "recovered successfully".
                    if recovery_result and recovery_result.get("status") in ("completed", "success"):
                        self.logger.info(f"Step '{step_name}' recovered successfully")
                        results[-1] = recovery_result  # replace failed result
                    elif step.get("critical", False):
                        self.logger.error(f"Critical step {step_name} failed and could not be recovered")
                    else:
                        self.logger.warning(f"Non-critical step {step_name} failed and could not be recovered, continuing workflow")
                else:
                    self.logger.info(f"Step {step_name} completed successfully")
            
            # Generate workflow report
            report = self._generate_workflow_report(workflow, results, output_dir)
            
            # Determine overall workflow status
            workflow_status = "success"
            for result in results:
                if result.get("status") in ("error", "failed"):
                    step_name = result.get("step_name", "unknown")
                    for step in workflow_steps:
                        if step.get("name") == step_name and step.get("critical", True):
                            workflow_status = "failed"
                            break
            
            # Generate workflow DAG visualization
            workflow_steps_with_status = []
            for i, step in enumerate(workflow_steps):
                # Check if step is a dictionary or a WorkflowStep object
                if isinstance(step, dict):
                    # Create a WorkflowStep from dictionary
                    step_copy = WorkflowStep(
                        name=step.get("name", f"Step {i+1}"),
                        description=step.get("description", ""),
                        command=step.get("command", ""),
                        dependencies=step.get("dependencies", [])
                    )
                else:
                    # Create a copy of the WorkflowStep object
                    step_copy = WorkflowStep(
                        name=step.name,
                        description=step.description,
                        command=step.command,
                        dependencies=step.dependencies
                    )
                
                # Set status based on execution results
                result_step = next((r for r in results if r.get("step_name") == step_copy.name), None)
                if result_step:
                    step_copy.status = result_step.get("status", "pending")
                else:
                    # For steps not found in results, check if they were skipped due to smart resume
                    # If the step name is in completed_steps (smart resume), mark it as completed
                    if hasattr(workflow, 'completed_steps') and step_copy.name in workflow.completed_steps:
                        step_copy.status = "completed"
                        self.logger.debug(f"Setting status of step {step_copy.name} to 'completed' based on smart resume")
                
                workflow_steps_with_status.append(step_copy)
            
            dag_image_path = self._save_workflow_dag(workflow_steps_with_status, output_dir)
            
            # Write workflow manifest for reproducibility
            try:
                self._write_manifest(output_dir, prompt, workflow_plan, results)
            except Exception as manifest_err:
                self.logger.warning("Failed to write manifest: %s", manifest_err)

            # Return results
            result_dict = {
                "status": workflow_status,
                "workflow": workflow,
                "results": results,
                "report": report,
                "output_dir": output_dir
            }
            
            # Only add dag_visualization if it was successfully created
            if dag_image_path:
                result_dict["dag_visualization"] = str(dag_image_path)
                
            return result_dict
        
        except Exception as e:
            self.logger.error(f"Error executing workflow: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    async def plan_workflow(self, prompt: str) -> Dict[str, Any]:
        """Plan a workflow based on a natural language prompt without executing it.
        
        Args:
            prompt: Natural language prompt describing the workflow
            
        Returns:
            dict: Workflow plan
        """
        try:
            # Plan the workflow
            self.logger.info("Planning workflow steps...")
            workflow_plan = await self.llm.generate_workflow_plan(prompt)
            
            # Check dependencies with a timeout
            self.logger.info("Checking dependencies...")
            try:
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Dependency checking timed out")
                
                # Set a 5-minute timeout for dependency checking (conda installs can be slow)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(300)
                
                try:
                    all_installed, available_but_failed_install = await self.dependency_manager.ensure_workflow_dependencies(workflow_plan)
                finally:
                    # Cancel the alarm
                    signal.alarm(0)
                
                if not all_installed:
                    if available_but_failed_install:
                        # Some dependencies failed to install but are available in the environment
                        missing_deps = [dep for dep in workflow_plan.get("dependencies", {}).get("tools", []) 
                                      if isinstance(dep, dict) and dep.get("name", "") not in available_but_failed_install
                                      or not isinstance(dep, dict) and dep not in available_but_failed_install]
                        if missing_deps:
                            self.logger.warning(f"Some dependencies could not be installed and are not available: {missing_deps}")
                            self.logger.warning("Continuing with workflow execution as some tools were found in the environment.")
                        else:
                            self.logger.info("All required tools are available in the environment despite installation failures.")
                    else:
                        # No dependencies are available, cannot proceed
                        self.logger.error("Dependencies not installed. Cannot execute workflow.")
                        return {"status": "error", "message": "Dependencies not installed. Cannot execute workflow."}
            
            except TimeoutError as e:
                self.logger.warning(f"Dependency checking timed out: {str(e)}")
                self.logger.warning("Continuing with workflow execution without full dependency verification")
            
            except Exception as e:
                self.logger.error(f"Error during dependency checking: {str(e)}")
                self.logger.warning("Continuing with workflow execution despite dependency checking error")
            
            return workflow_plan
            
        except Exception as e:
            self.logger.error(f"Failed to plan workflow: {str(e)}")
            raise

    async def analyze_results(self, workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workflow results using agentic system."""
        try:
            self.logger.info("Starting agentic analysis of workflow results...")
            
            # Get results directory from workflow results
            results_dir = None
            
            # Method 1: Try primary output directory
            if "primary_output_dir" in workflow_results:
                results_dir = Path(workflow_results["primary_output_dir"])
            
            # Method 2: Try output directories list
            if not results_dir and "output_directories" in workflow_results:
                output_dirs = workflow_results["output_directories"]
                if output_dirs:
                    results_dir = Path(output_dirs[0])
            
            # Method 3: Try output_directory field
            if not results_dir:
                results_dir = Path(workflow_results.get("output_directory", "results"))
            
            # Validate directory exists
            if not results_dir or not results_dir.exists():
                self.logger.warning("No valid output directory found for analysis")
                return {
                    "status": "error",
                    "error": "No valid output directory found"
                }
                
            self.logger.info(f"Analyzing results in directory: {results_dir}")
            analysis_data = await self.analysis_system._prepare_analysis_data(results_dir)
            
            # Run analysis agents
            quality_analysis = await self.analysis_system.quality_agent.analyze(analysis_data)
            quant_analysis = await self.analysis_system.quantification_agent.analyze(analysis_data)
            tech_analysis = await self.analysis_system.technical_agent.analyze(analysis_data)
            
            self.logger.info("Agentic analysis completed successfully")
            
            return {
                "status": "success",
                "quality": quality_analysis,
                "quantification": quant_analysis,
                "technical": tech_analysis,
                "data": analysis_data
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def resume_workflow(self, prompt: str, checkpoint_dir: str, force_resume: bool = False) -> Dict[str, Any]:
        """Resume workflow execution from checkpoint.
        
        Args:
            prompt: Natural language prompt describing the workflow
            checkpoint_dir: Directory to load workflow checkpoint from
            force_resume: If True, skip smart resume and run all steps
            
        Returns:
            dict: Workflow execution results
        """
        try:
            self.logger.info(f"Resuming workflow from checkpoint: {checkpoint_dir}")
            
            # Load checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.json")
            if not os.path.exists(checkpoint_path):
                self.logger.error(f"Checkpoint file not found: {checkpoint_path}")
                return None
            
            try:
                with open(checkpoint_path, "r") as f:
                    checkpoint = json.load(f)
                
                workflow_plan = checkpoint.get("workflow_plan", {})
                output_dir = checkpoint.get("output_dir", os.path.abspath("results"))
                
                # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)
                
                # Detect completed steps only if force_resume is False
                completed_steps = set()
                if not force_resume:
                    # Convert to step_dicts format for smart resume, preserving
                    # the planner's ``outputs`` list so generic_validator can
                    # check declared artifacts rather than re-parsing the shell.
                    step_dicts = []
                    for step in workflow_plan.get("steps", []):
                        step_dict = {
                            "name": step.get("name", ""),
                            "command": step.get("command", ""),
                            "description": step.get("description", ""),
                            "dependencies": step.get("dependencies", []),
                            "outputs": step.get("outputs", []),
                        }
                        step_dicts.append(step_dict)
                    
                    # Detect completed steps
                    from flowagent.core.smart_resume import detect_completed_steps
                    completed_steps = detect_completed_steps(step_dicts)
                    if completed_steps:
                        self.logger.info(f"Detected {len(completed_steps)} completed steps: {', '.join(completed_steps)}")
                    else:
                        self.logger.info("No completed steps detected, starting from the beginning")
                else:
                    self.logger.info("Force resume enabled, running all steps regardless of completion status")
            
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint: {str(e)}")
                return None
            
            # Get workflow name and description
            workflow_name = workflow_plan.get("name", "Unnamed workflow")
            workflow_description = workflow_plan.get("description", "")
            
            # Create workflow object
            workflow = Workflow(
                name=workflow_name,
                description=workflow_description,
                steps=[],
                dependencies=workflow_plan.get("dependencies", {}).get("tools", []),
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir
            )
            
            # Create steps
            for step_data in workflow_plan.get("steps", []):
                # Get resource requirements for the step
                resource_requirements = self._get_resource_requirements(step_data.get("tool", ""))
                
                # Create step object
                step = WorkflowStep(
                    name=step_data.get("name", ""),
                    description=step_data.get("description", ""),
                    tool=step_data.get("tool", ""),
                    command=step_data.get("command", ""),
                    args=step_data.get("args", {}),
                    dependencies=step_data.get("dependencies", []),
                    output_files=step_data.get("output_files", []),
                    critical=step_data.get("critical", True),
                    resource_requirements=resource_requirements
                )
                
                workflow.steps.append(step)
            
            # If a checkpoint directory is provided, use smart resume functionality
            if checkpoint_dir:
                self.logger.info("Using smart resume functionality")
                # Convert WorkflowStep objects to dictionaries for smart resume.
                # Pull the planner's ``outputs`` list through by name — it's the
                # authoritative source of expected artifacts and makes
                # completion detection reliable for `cd`-prefixed commands,
                # pipelines, and xargs-based downloads where output paths
                # can't be parsed from the shell text.
                plan_by_name = {
                    s.get("name"): s for s in workflow_plan.get("steps", [])
                } if isinstance(workflow_plan, dict) else {}
                step_dicts = []
                for step in workflow.steps:
                    plan_step = plan_by_name.get(step.name, {})
                    step_dict = {
                        "name": step.name,
                        "command": step.command,
                        "dependencies": step.dependencies,
                        "outputs": plan_step.get("outputs", []),
                    }
                    step_dicts.append(step_dict)
                
                # Detect completed steps
                completed_steps = detect_completed_steps(step_dicts)
                if completed_steps:
                    self.logger.info(f"Detected {len(completed_steps)} completed steps: {', '.join(completed_steps)}")
                
                # Store completed steps in the workflow object for later use in DAG visualization
                workflow.completed_steps = completed_steps
                
                # Filter workflow steps
                filtered_steps = filter_workflow_steps(step_dicts, completed_steps)

                # Update workflow steps based on filtered steps
                if len(filtered_steps) < len(step_dicts):
                    self.logger.info(f"Filtered {len(step_dicts) - len(filtered_steps)} steps, will run {len(filtered_steps)} steps")
                    # Keep only the steps that are in the filtered list
                    filtered_names = {fs["name"] for fs in filtered_steps}
                    workflow.steps = [s for s in workflow.steps if s.name in filtered_names]
                    # Propagate scrubbed dependency lists onto the surviving
                    # WorkflowStep objects so the executor's DAG build doesn't
                    # reject them for referencing filtered-out completed steps.
                    scrubbed_deps_by_name = {fs["name"]: fs.get("dependencies", []) for fs in filtered_steps}
                    for s in workflow.steps:
                        if s.name in scrubbed_deps_by_name:
                            s.dependencies = scrubbed_deps_by_name[s.name]
            
            # Execute the workflow
            result = await self.execute_workflow(workflow)
            
            # Generate analysis report if workflow was successful
            if result.get("status") == "success":
                self.logger.info("Workflow completed successfully")
                
                # Generate a report if requested
                if workflow_plan.get("generate_report", False):
                    self.logger.info("Generating report...")
                    from ..analysis.report_generator import ReportGenerator
                    report_generator = ReportGenerator()
                    report_path = await report_generator.generate_report(workflow_plan, result)
                    result["analysis_report"] = report_path
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to resume workflow: {str(e)}")
            raise

    async def _prepare_step(self, step: Dict[str, Any], workflow_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a workflow step for execution."""
        try:
            # Extract basic step info
            step_name = step.get("name", "unknown")
            command = step.get("command", "")
            dependencies = step.get("dependencies", [])
            
            # Get resource requirements
            resources = step.get("resources", {})
            profile = resources.get("profile", "default")
            
            # Create the step dictionary
            prepared_step = {
                "name": step_name,
                "command": command,
                "status": "pending",
                "dependencies": dependencies,
                "profile": profile,
                "output": "",
                "error": None,
                "start_time": None,
                "end_time": None
            }
            
            return prepared_step
            
        except Exception as e:
            self.logger.error(f"Error preparing step {step.get('name', 'unknown')}: {str(e)}")
            raise

    def _build_dag(self, workflow_steps: List[WorkflowStep]) -> nx.DiGraph:
        """Build DAG from workflow steps."""
        dag = nx.DiGraph()
        for step in workflow_steps:
            dag.add_node(step.name, step=step, status='pending')
            for dep in step.dependencies:
                dag.add_edge(dep, step.name)
        return dag

    def _save_workflow_dag(self, workflow_steps: List[WorkflowStep], output_dir: str) -> Optional[Path]:
        """Generate and save workflow DAG visualization.
        
        Args:
            workflow_steps: List of workflow steps
            output_dir: Directory to save the DAG image
            
        Returns:
            Path to the saved DAG image, or None if visualization failed
        """
        try:
            # Check if output_dir is None
            if output_dir is None:
                self.logger.warning("No output directory specified for workflow DAG visualization")
                return None
                
            # Check if workflow_steps is None or empty
            if not workflow_steps:
                self.logger.warning("No workflow steps provided for DAG visualization")
                return None
                
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            # Build the DAG
            try:
                dag = self._build_dag(workflow_steps)
                if not dag or not dag.nodes:
                    self.logger.warning("Built DAG is empty, nothing to visualize")
                    return None
            except Exception as e:
                self.logger.error(f"Error building DAG: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                return None
            
            # Update status of each step in the DAG based on workflow results
            for step in workflow_steps:
                try:
                    if step.name in dag.nodes:
                        # Get status from step, defaulting to 'pending'
                        status = 'pending'
                        if hasattr(step, 'status'):
                            status = step.status
                        elif isinstance(step, dict) and 'status' in step:
                            status = step['status']
                        
                        # Update the node's step status
                        # Convert WorkflowStep objects to dictionaries for compatibility with the visualizer
                        if isinstance(dag.nodes[step.name]['step'], dict):
                            dag.nodes[step.name]['step']['status'] = status
                        else:
                            # Convert WorkflowStep to dict if it's not already
                            step_as_dict = {
                                'name': dag.nodes[step.name]['step'].name,
                                'status': status,
                                'command': dag.nodes[step.name]['step'].command if hasattr(dag.nodes[step.name]['step'], 'command') else '',
                                'dependencies': dag.nodes[step.name]['step'].dependencies if hasattr(dag.nodes[step.name]['step'], 'dependencies') else []
                            }
                            dag.nodes[step.name]['step'] = step_as_dict
                except Exception as e:
                    self.logger.warning(f"Error updating step {getattr(step, 'name', 'unknown')} in DAG: {str(e)}")
                    continue
            
            # Create WorkflowDAG instance and visualize
            try:
                workflow_dag = WorkflowDAG(executor_type=self.executor_type)
                workflow_dag.graph = dag
                
                # Save visualization to output directory
                dag_image_path = Path(os.path.join(output_dir, "workflow_dag.png"))
                result_path = workflow_dag.visualize(dag_image_path)
                
                if result_path:
                    self.logger.info(f"Saved workflow DAG visualization to {result_path}")
                    return result_path
                else:
                    self.logger.warning("Failed to save workflow DAG visualization")
                    return None
            except Exception as e:
                self.logger.error(f"Error visualizing DAG: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                return None
        except Exception as e:
            self.logger.error(f"Failed to save workflow DAG visualization: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def get_execution_plan(self, dag: nx.DiGraph) -> list:
        """Get ordered list of task batches that can be executed in parallel."""
        execution_plan = []
        remaining_nodes = set(dag.nodes())
        
        while remaining_nodes:
            # Find nodes with no incomplete dependencies
            ready_nodes = {
                node for node in remaining_nodes
                if not any(pred in remaining_nodes for pred in dag.predecessors(node))
            }
            
            if not ready_nodes:
                # There are nodes left but none are ready - there must be a cycle
                raise ValueError("Cycle detected in workflow DAG")
            
            execution_plan.append(list(ready_nodes))
            remaining_nodes -= ready_nodes
        
        return execution_plan

    async def _generate_analysis_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate analysis report from workflow results."""
        try:
            # Check if we have any results
            if not results:
                return {
                    "status": None,
                    "quality": "unknown",
                    "issues": [{
                        "severity": "high",
                        "description": "No tool outputs available",
                        "impact": "Lack of tool outputs makes it impossible to assess the quality or draw any meaningful conclusions from the analysis.",
                        "solution": "Check the workflow configuration to ensure that tools are properly executed and their outputs are captured for analysis."
                    }],
                    "warnings": [{
                        "severity": "high",
                        "description": "Missing tool outputs",
                        "impact": "Without tool outputs, it is not possible to verify the analysis results or troubleshoot any potential issues.",
                        "solution": "Review the workflow execution logs to identify any errors or issues that might have prevented tool outputs from being generated."
                    }],
                    "recommendations": [{
                        "type": "quality",
                        "description": "Ensure all tools in the workflow are properly configured and executed to generate necessary outputs.",
                        "reason": "Having complete tool outputs is essential for quality assessment and interpretation of the analysis results."
                    }]
                }
            
            # Analyze results
            status = all(r["status"] == "success" for r in results)
            quality = "good" if status else "poor"
            
            issues = []
            warnings = []
            recommendations = []
            
            for result in results:
                if result["status"] != "success":
                    issues.append({
                        "severity": "high",
                        "description": f"Step '{result['step']}' failed",
                        "error": result.get("error", "Unknown error"),
                        "diagnosis": result.get("diagnosis", {})
                    })
            
            return {
                "status": "success" if status else "failed",
                "quality": quality,
                "issues": issues,
                "warnings": warnings,
                "recommendations": recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate analysis report: {str(e)}")
            return None

    def _generate_workflow_report(self, workflow: Dict[str, Any], results: List[Dict[str, Any]], output_dir: str) -> Dict[str, Any]:
        """Generate a report for the workflow execution."""
        # Get workflow name and description
        workflow_name = workflow.get("name", "Unnamed workflow")
        workflow_description = workflow.get("description", "")
        
        # Calculate workflow statistics
        total_steps = len(results)
        successful_steps = sum(1 for r in results if r.get("status") == "success")
        failed_steps = sum(1 for r in results if r.get("status") == "error")
        skipped_steps = sum(1 for r in results if r.get("status") == "skipped")
        
        # Calculate execution time
        total_execution_time = sum(r.get("execution_time", 0) for r in results if "execution_time" in r)
        
        # Get dependency check results
        all_tools = set()
        for step in workflow.get("steps", []):
            all_tools.update(step.get("tools", []))
        
        dependency_results = self.dependency_manager.check_dependencies(workflow.get("steps", []))
        available_tools = dependency_results.get("available", [])
        missing_tools = dependency_results.get("missing", [])
        
        # Generate step reports
        step_reports = []
        for i, (step, result) in enumerate(zip(workflow.get("steps", []), results)):
            step_name = step.get("name", f"Step {i+1}")
            step_status = result.get("status", "unknown")
            step_execution_time = result.get("execution_time", 0)
            
            # Get tools for this step
            step_tools = step.get("tools", [])
            missing_step_tools = [tool for tool in step_tools if tool in missing_tools]
            
            step_report = {
                "name": step_name,
                "status": step_status,
                "execution_time": step_execution_time,
                "command": step.get("command", ""),
                "tools": step_tools,
                "missing_tools": missing_step_tools,
                "output": result.get("stdout", ""),
                "error": result.get("stderr", ""),
                "critical": step.get("critical", False)
            }
            
            step_reports.append(step_report)
        
        # Generate report
        report = {
            "workflow_name": workflow_name,
            "workflow_description": workflow_description,
            "execution_summary": {
                "total_steps": total_steps,
                "successful_steps": successful_steps,
                "failed_steps": failed_steps,
                "skipped_steps": skipped_steps,
                "total_execution_time": total_execution_time
            },
            "dependency_summary": {
                "total_tools": len(all_tools),
                "available_tools": available_tools,
                "missing_tools": missing_tools,
                "availability_percentage": len(available_tools) / len(all_tools) * 100 if all_tools else 100
            },
            "steps": step_reports,
            "output_directory": output_dir
        }
        
        # Save report to output directory
        report_file = os.path.join(output_dir, "report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        # Generate HTML report
        html_report = self._generate_html_report(report)
        html_report_file = os.path.join(output_dir, "report.html")
        with open(html_report_file, "w") as f:
            f.write(html_report)
        
        report["report_file"] = report_file
        report["html_report_file"] = html_report_file
        
        return report

    async def plan_and_execute_workflow(
        self, prompt, output_dir=None, checkpoint_dir=None, workflow_plan=None,
    ):
        """Plan and execute a workflow based on a natural language prompt.

        Args:
            prompt: Natural language prompt describing the workflow
            output_dir: Directory to store workflow outputs
            checkpoint_dir: Directory to store workflow checkpoints
            workflow_plan: Pre-computed plan from a prior ``plan_workflow`` call.
                When provided, planning and dependency-installation are skipped
                so the same run doesn't re-plan / re-install.

        Returns:
            dict: Workflow execution results
        """
        try:
            self.logger.info(f"Initial working directory: {os.getcwd()}")
            self.logger.info(f"Using {self.executor_type} executor")

            if workflow_plan is None:
                self.logger.info("Planning workflow steps...")
                workflow_plan = await self.llm.generate_workflow_plan(prompt)

                self.logger.info("Checking dependencies...")
                all_installed, available_but_failed_install = await self.dependency_manager.ensure_workflow_dependencies(workflow_plan)
            else:
                self.logger.info("Reusing pre-computed workflow plan (skipping re-plan and dependency re-install)")
                all_installed, available_but_failed_install = True, []
            
            if not all_installed:
                if available_but_failed_install:
                    # Some dependencies failed to install but are available in the environment
                    missing_deps = [dep for dep in workflow_plan.get("dependencies", {}).get("tools", []) if dep not in available_but_failed_install]
                    if missing_deps:
                        self.logger.warning(f"Some dependencies could not be installed and are not available: {missing_deps}")
                        self.logger.warning("Continuing with workflow execution as some tools were found in the environment.")
                    else:
                        self.logger.info("All required tools are available in the environment despite installation failures.")
                else:
                    # No dependencies are available, cannot proceed
                    self.logger.error("Dependencies not installed. Cannot execute workflow.")
                    return {"status": "error", "message": "Dependencies not installed. Cannot execute workflow."}
            
            # Get output directory from workflow steps
            if output_dir is None:
                for step in workflow_plan.get("steps", []):
                    if "output_dir" in step:
                        output_dir = step["output_dir"]
                        break
            
            # Create workflow object
            workflow = Workflow(
                name=workflow_plan.get("name", "Unnamed Workflow"),
                description=workflow_plan.get("description", ""),
                steps=[],
                dependencies=workflow_plan.get("dependencies", {}).get("tools", []),
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir
            )
            
            # Create steps
            for step_data in workflow_plan.get("steps", []):
                # Get resource requirements for the step
                resource_requirements = self._get_resource_requirements(step_data.get("tool", ""))
                
                # Create step object
                step = WorkflowStep(
                    name=step_data.get("name", ""),
                    description=step_data.get("description", ""),
                    tool=step_data.get("tool", ""),
                    command=step_data.get("command", ""),
                    args=step_data.get("args", {}),
                    dependencies=step_data.get("dependencies", []),
                    output_files=step_data.get("output_files", []),
                    critical=step_data.get("critical", True),
                    resource_requirements=resource_requirements
                )
                
                workflow.steps.append(step)
            
            # If a checkpoint directory is provided, use smart resume functionality
            if checkpoint_dir:
                self.logger.info("Using smart resume functionality")
                # Convert WorkflowStep objects to dictionaries for smart resume.
                # Pull the planner's ``outputs`` list through by name — it's the
                # authoritative source of expected artifacts and makes
                # completion detection reliable for `cd`-prefixed commands,
                # pipelines, and xargs-based downloads where output paths
                # can't be parsed from the shell text.
                plan_by_name = {
                    s.get("name"): s for s in workflow_plan.get("steps", [])
                } if isinstance(workflow_plan, dict) else {}
                step_dicts = []
                for step in workflow.steps:
                    plan_step = plan_by_name.get(step.name, {})
                    step_dict = {
                        "name": step.name,
                        "command": step.command,
                        "dependencies": step.dependencies,
                        "outputs": plan_step.get("outputs", []),
                    }
                    step_dicts.append(step_dict)
                
                # Detect completed steps
                completed_steps = detect_completed_steps(step_dicts)
                if completed_steps:
                    self.logger.info(f"Detected {len(completed_steps)} completed steps: {', '.join(completed_steps)}")
                
                # Store completed steps in the workflow object for later use in DAG visualization
                workflow.completed_steps = completed_steps
                
                # Filter workflow steps
                filtered_steps = filter_workflow_steps(step_dicts, completed_steps)

                # Update workflow steps based on filtered steps
                if len(filtered_steps) < len(step_dicts):
                    self.logger.info(f"Filtered {len(step_dicts) - len(filtered_steps)} steps, will run {len(filtered_steps)} steps")
                    # Keep only the steps that are in the filtered list
                    filtered_names = {fs["name"] for fs in filtered_steps}
                    workflow.steps = [s for s in workflow.steps if s.name in filtered_names]
                    # Propagate scrubbed dependency lists onto the surviving
                    # WorkflowStep objects so the executor's DAG build doesn't
                    # reject them for referencing filtered-out completed steps.
                    scrubbed_deps_by_name = {fs["name"]: fs.get("dependencies", []) for fs in filtered_steps}
                    for s in workflow.steps:
                        if s.name in scrubbed_deps_by_name:
                            s.dependencies = scrubbed_deps_by_name[s.name]
            
            # Execute the workflow
            result = await self.execute_workflow(workflow)
            
            # Generate analysis report if workflow was successful
            if result.get("status") == "success":
                self.logger.info("Workflow completed successfully")
                
                # Generate a report if requested
                if workflow_plan.get("generate_report", False):
                    self.logger.info("Generating report...")
                    from ..analysis.report_generator import ReportGenerator
                    report_generator = ReportGenerator()
                    report_path = await report_generator.generate_report(workflow_plan, result)
                    result["analysis_report"] = report_path
            
            # Generate workflow DAG visualization
            workflow_steps_with_status = []
            for i, step in enumerate(workflow.steps):
                # Check if step is a dictionary or a WorkflowStep object
                if isinstance(step, dict):
                    # Create a WorkflowStep from dictionary
                    step_copy = WorkflowStep(
                        name=step.get("name", f"Step {i+1}"),
                        description=step.get("description", ""),
                        command=step.get("command", ""),
                        dependencies=step.get("dependencies", [])
                    )
                else:
                    # Create a copy of the WorkflowStep object
                    step_copy = WorkflowStep(
                        name=step.name,
                        description=step.description,
                        command=step.command,
                        dependencies=step.dependencies
                    )
                
                # Set status based on execution results
                result_step = next((r for r in result.get("results", []) if r.get("step_name") == step_copy.name), None)
                if result_step:
                    step_copy.status = result_step.get("status", "pending")
                else:
                    # For steps not found in results, check if they were skipped due to smart resume
                    # If the step name is in completed_steps (smart resume), mark it as completed
                    if hasattr(workflow, 'completed_steps') and step_copy.name in workflow.completed_steps:
                        step_copy.status = "completed"
                        self.logger.debug(f"Setting status of step {step_copy.name} to 'completed' based on smart resume")
                
                workflow_steps_with_status.append(step_copy)
            
            dag_image_path = self._save_workflow_dag(workflow_steps_with_status, output_dir)
            
            # Only add dag_visualization if it was successfully created
            if dag_image_path:
                result["dag_visualization"] = str(dag_image_path)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    def _extract_geo_accession(self, prompt: str) -> Optional[str]:
        """Extract GEO accession number from prompt.
        
        Args:
            prompt: User prompt
            
        Returns:
            GEO accession number or None if not found
        """
        import re
        
        # Define regex pattern for GEO accession numbers
        geo_pattern = r'(?:GSE|GDS|GSM)\d+'
        
        # Search for GEO accession numbers in the prompt
        match = re.search(geo_pattern, prompt)
        
        if match:
            return match.group(0)
        
        return None
    
    def _add_geo_download_steps(self, geo_accession: str, workflow_steps: List[Dict[str, Any]], output_dir: str) -> List[Dict[str, Any]]:
        """Add GEO download steps to workflow.
        
        Args:
            geo_accession: GEO accession number
            workflow_steps: Original workflow steps
            output_dir: Output directory for the workflow
            
        Returns:
            Updated workflow steps with GEO download steps added
        """
        # Create data directory
        data_dir = os.path.join(output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Define GEO download steps
        geo_steps = [
            {
                "name": f"Get SRR IDs for {geo_accession}",
                "description": f"Retrieve SRR IDs for GEO accession {geo_accession}",
                "command": f"esearch -db sra -query '{geo_accession}[Accession]' | efetch -format runinfo > {data_dir}/{geo_accession}_runinfo.csv",
                "tools": ["esearch", "efetch"],
                "critical": True
            },
            {
                "name": f"Extract SRR IDs from {geo_accession} runinfo",
                "description": "Extract SRR IDs from runinfo CSV file",
                "command": f"tail -n +2 {data_dir}/{geo_accession}_runinfo.csv | cut -d',' -f1 > {data_dir}/{geo_accession}_srr_ids.txt",
                "tools": ["tail", "cut"],
                "critical": True
            },
            {
                "name": f"Download SRA files for {geo_accession}",
                "description": "Download SRA files using prefetch",
                "command": f"cat {data_dir}/{geo_accession}_srr_ids.txt | xargs -I{{}} prefetch {{}} --output-directory {data_dir}",
                "tools": ["prefetch"],
                "critical": True
            },
            {
                "name": f"Convert SRA to FASTQ for {geo_accession}",
                "description": "Convert SRA files to FASTQ format",
                "command": f"cat {data_dir}/{geo_accession}_srr_ids.txt | xargs -I{{}} fasterq-dump {{}} -O {data_dir}",
                "tools": ["fasterq-dump"],
                "critical": True
            },
            {
                "name": f"Compress FASTQ files for {geo_accession}",
                "description": "Compress FASTQ files with gzip",
                "command": f"find {data_dir} -name '*.fastq' -exec gzip {{}} \\;",
                "tools": ["gzip", "find"],
                "critical": False
            }
        ]
        
        # Add GEO download steps to the beginning of the workflow
        return geo_steps + workflow_steps
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report from workflow report."""
        # Generate HTML report
        html_report = "<html><body>"
        html_report += "<h1>Workflow Report</h1>"
        html_report += "<h2>Workflow Summary</h2>"
        html_report += f"<p>Workflow Name: {report['workflow_name']}</p>"
        html_report += f"<p>Workflow Description: {report['workflow_description']}</p>"
        html_report += "<h2>Execution Summary</h2>"
        html_report += f"<p>Total Steps: {report['execution_summary']['total_steps']}</p>"
        html_report += f"<p>Successful Steps: {report['execution_summary']['successful_steps']}</p>"
        html_report += f"<p>Failed Steps: {report['execution_summary']['failed_steps']}</p>"
        html_report += f"<p>Skipped Steps: {report['execution_summary']['skipped_steps']}</p>"
        html_report += f"<p>Total Execution Time: {report['execution_summary']['total_execution_time']} seconds</p>"
        html_report += "<h2>Dependency Summary</h2>"
        html_report += f"<p>Total Tools: {report['dependency_summary']['total_tools']}</p>"
        html_report += f"<p>Available Tools: {', '.join(report['dependency_summary']['available_tools'])}</p>"
        html_report += f"<p>Missing Tools: {', '.join(report['dependency_summary']['missing_tools'])}</p>"
        html_report += f"<p>Availability Percentage: {report['dependency_summary']['availability_percentage']}%</p>"
        html_report += "<h2>Step Reports</h2>"
        for step in report["steps"]:
            html_report += f"<h3>Step {step['name']}</h3>"
            html_report += f"<p>Status: {step['status']}</p>"
            html_report += f"<p>Execution Time: {step['execution_time']} seconds</p>"
            html_report += f"<p>Command: {step['command']}</p>"
            html_report += f"<p>Tools: {', '.join(step['tools'])}</p>"
            html_report += f"<p>Missing Tools: {', '.join(step['missing_tools'])}</p>"
            html_report += f"<p>Output: {step['output']}</p>"
            html_report += f"<p>Error: {step['error']}</p>"
            html_report += f"<p>Critical: {step['critical']}</p>"
        html_report += "</body></html>"
        
        return html_report

    def _get_resource_requirements(self, tool_name: str) -> Dict[str, Any]:
        """Get resource requirements for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Dictionary of resource requirements
        """
        # Default resource requirements
        default_requirements = {
            "cpu": 1,
            "memory": "1G",
            "time": "1:00:00"
        }
        
        # Tool-specific resource requirements
        tool_requirements = {
            # RNA-seq tools
            "star": {"cpu": 8, "memory": "32G", "time": "4:00:00"},
            "kallisto": {"cpu": 4, "memory": "8G", "time": "2:00:00"},
            "salmon": {"cpu": 4, "memory": "8G", "time": "2:00:00"},
            "stringtie": {"cpu": 4, "memory": "8G", "time": "2:00:00"},
            
            # Alignment tools
            "bwa": {"cpu": 8, "memory": "16G", "time": "4:00:00"},
            "bowtie": {"cpu": 4, "memory": "8G", "time": "2:00:00"},
            "bowtie2": {"cpu": 4, "memory": "8G", "time": "2:00:00"},
            "hisat2": {"cpu": 4, "memory": "8G", "time": "2:00:00"},
            
            # Variant calling tools
            "gatk": {"cpu": 4, "memory": "16G", "time": "4:00:00"},
            "samtools": {"cpu": 2, "memory": "4G", "time": "2:00:00"},
            "bcftools": {"cpu": 2, "memory": "4G", "time": "2:00:00"},
            
            # QC tools
            "fastqc": {"cpu": 1, "memory": "2G", "time": "1:00:00"},
            "multiqc": {"cpu": 1, "memory": "2G", "time": "0:30:00"},
            
            # SRA tools
            "prefetch": {"cpu": 1, "memory": "2G", "time": "4:00:00"},
            "fasterq-dump": {"cpu": 4, "memory": "8G", "time": "4:00:00"},
            "fastq-dump": {"cpu": 2, "memory": "4G", "time": "4:00:00"},
            
            # Single-cell tools
            "cellranger": {"cpu": 16, "memory": "64G", "time": "24:00:00"},
            "kb": {"cpu": 8, "memory": "32G", "time": "8:00:00"},
            "velocyto": {"cpu": 8, "memory": "32G", "time": "8:00:00"}
        }
        
        # Convert tool name to lowercase for case-insensitive matching
        tool_name_lower = tool_name.lower()
        
        # Return tool-specific requirements or default
        return tool_requirements.get(tool_name_lower, default_requirements)
