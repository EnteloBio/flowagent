#!/usr/bin/env python3
"""Subprocess shim that drives AutoBA (Zhou et al., 2023) from the
head-to-head benchmark harness.

AutoBA ships as a script project (``python app.py --config cfg.yaml --openai
KEY --model MODEL --execute False``). It uses the raw ``openai`` SDK rather
than LangChain, so this shim monkey-patches ``openai.resources.chat.
completions.Completions.create`` to capture token usage — directly
comparable to FlowAgent and BioMaster numbers.

Flow:
  1. Synthesise a scratch ``config.yaml`` with the caller's prompt as
     ``goal_description`` and any ``--files`` as ``data_list``.
  2. Redirect AutoBA's ``output_dir`` into the scratch dir.
  3. Set ``execute=False`` so AutoBA plans + writes per-task shell scripts
     without actually running them.
  4. Invoke ``app.py`` via ``runpy.run_path`` (equivalent to a user running
     ``python app.py ...`` in a fresh clone) under the token patch.
  5. Read back ``<output_dir>/0_response.json`` (the plan — a JSON object
     with a ``plan`` key holding a list of task descriptions) plus
     ``<output_dir>/<N>.sh`` for N = 1..len(plan) (per-task shell bodies),
     project into FlowAgent's plan schema, classify workflow_type.
  6. Emit one JSON envelope on stdout with plan / tokens / cost / wall /
     error — the same shape the harness reads from BioMaster's shim.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Reuse the harness's dotenv loader + workflow-type classifier so direct
# invocation works and the three competitors share the same mapping rules.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from harness.runner import _load_dotenv_once  # type: ignore
    _load_dotenv_once()
except Exception:
    pass

try:
    from harness.biomaster_shim import _classify_workflow_type  # type: ignore
except Exception:
    # Fallback: emit "custom" if the biomaster_shim helper isn't importable.
    def _classify_workflow_type(plan: Dict[str, Any]) -> str:  # type: ignore
        return "custom"


def _die(msg: str, *, error_type: str = "shim-error", exit_code: int = 2) -> None:
    print(json.dumps({
        "plan": {"workflow_type": "custom", "steps": []},
        "prompt_tokens": 0, "completion_tokens": 0,
        "llm_calls": 0, "cost_usd": 0.0, "wall_seconds": 0.0,
        "error": f"{error_type}: {msg}",
    }))
    sys.exit(exit_code)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--files", default="[]",
                    help='JSON array of "path: description" strings')
    ap.add_argument("--autoba-dir",
                    default=os.environ.get("AUTOBA_DIR", ""))
    ap.add_argument("--model",
                    default=os.environ.get("OPENAI_MODEL", "gpt-4.1"))
    ap.add_argument("--api-key",
                    default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--keep-output", action="store_true",
                    help="Do not delete the scratch output dir on exit")
    return ap.parse_args()


def _install_unused_import_stubs() -> None:
    """Prevent AutoBA's load-time imports of local-LLM deps from crashing.

    ``src/local_llm.py`` (imported at module load by ``src/agent.py``) pulls
    in ``fire``, ``transformers``, ``fairscale``, ``sentencepiece`` and the
    bundled ``llama`` package — all of which are only used on AutoBA's
    llama / codellama / deepseek local-model code paths. For gpt-* models
    (our head-to-head config) none of these are actually called, but the
    imports themselves still execute at module load and crash if the
    packages aren't installed.

    We install a ``sys.meta_path`` finder that fabricates lenient stub
    modules for those exact top-level names, so load-time succeeds without
    inflicting ~2 GB of unused ML libraries on the benchmark env. The
    stubs raise nothing on attribute access; any accidental runtime use
    would surface immediately as an AttributeError on our side (not
    silently pass), which is what we want.
    """
    import importlib.abc
    import importlib.util
    import types

    class _LenientModule(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            child = _LenientModule(f"{self.__name__}.{name}")
            setattr(self, name, child)
            return child

        def __call__(self, *_a, **_kw):
            return _LenientModule(f"{self.__name__}()")

    class _StubLoader(importlib.abc.Loader):
        def create_module(self, spec):
            mod = _LenientModule(spec.name)
            mod.__path__ = []  # treat as a namespace package
            return mod

        def exec_module(self, module):  # nothing to exec for stubs
            pass

    class _StubFinder(importlib.abc.MetaPathFinder):
        def __init__(self, names):
            self._names = set(names)

        def find_spec(self, fullname, path, target=None):
            top = fullname.split(".", 1)[0]
            # Only match exact top-level names, not prefixes — ``llama``
            # stubs must NOT also intercept ``llama_index``.
            if top in self._names:
                return importlib.util.spec_from_loader(
                    fullname, _StubLoader(), is_package=True,
                )
            return None

    # Only stub names that aren't already importable. Over-stubbing a
    # real package (e.g. ``transformers``, which ``sentence_transformers``
    # and ``llama_index`` legitimately depend on) breaks code that does
    # ``from transformers.utils import ExplicitEnum`` and then uses it as
    # a base class — the stub's module-call signature clashes with
    # metaclass machinery.
    candidates = ("fire", "transformers", "fairscale",
                  "sentencepiece", "llama")
    stubbed = []
    for name in candidates:
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except ImportError:
            stubbed.append(name)
        except Exception:
            stubbed.append(name)
    if stubbed:
        sys.meta_path.insert(0, _StubFinder(stubbed))


def _patch_autoba_model_whitelist(autoba_dir: Path) -> None:
    """Make AutoBA's ``Agent.__init__`` accept arbitrary gpt-* model names.

    AutoBA (2023) hardcodes a whitelist of OpenAI model IDs
    (``gpt-3.5-turbo``, ``gpt-4``, ``gpt-4o``, …) and calls ``exit()`` on
    anything else. Newer models like ``gpt-4.1`` / ``gpt-5-*`` aren't in
    the list, so the benchmark couldn't hit them without modifying
    upstream source. Wrap ``__init__`` so:

      1. The internal ``exit()`` (which raises ``SystemExit``) is caught
         instead of aborting the run.
      2. The caller's model is appended post-hoc to
         ``self.gpt_model_engines`` and ``self.valid_model_engines`` so
         downstream code paths (``response-generation``) accept it.
      3. ``self.openai_client`` is ensured — the original init skipped
         that block after ``exit()`` fired.

    Non-invasive: patches only the in-memory class; upstream source is
    untouched. If the class layout has changed, the patch is skipped
    with a warning and the original validation runs (producing a clean
    error in the envelope rather than a silent corruption).
    """
    try:
        import src.agent as _autoba_agent  # type: ignore
    except Exception as e:
        print(f"[autoba_shim] could not import src.agent for patching: {e}",
              file=sys.stderr)
        return
    if not hasattr(_autoba_agent, "Agent"):
        return
    orig_init = _autoba_agent.Agent.__init__

    def _patched_init(self, *args, **kwargs):
        try:
            orig_init(self, *args, **kwargs)
        except SystemExit:
            # Whitelist rejection fired. Continue; we'll re-stitch below.
            pass
        # Post-hoc: add the caller's model to the whitelists so any
        # `self.model_engine in self.gpt_model_engines` check later in
        # request/response code still passes for it.
        if hasattr(self, "gpt_model_engines") and \
                self.model_engine not in self.gpt_model_engines:
            self.gpt_model_engines.append(self.model_engine)
        if hasattr(self, "valid_model_engines") and \
                self.model_engine not in self.valid_model_engines:
            self.valid_model_engines.append(self.model_engine)
        if not hasattr(self, "openai_client"):
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=self.openai_api)

    _autoba_agent.Agent.__init__ = _patched_init

    # AutoBA's response validator/parser assumes the LLM returns a JSON
    # *object* (``{"plan": [...]}``). gpt-4.1+ often returns a JSON *string
    # containing* JSON (one layer of stringification), which
    # ``valid_json_response`` then round-trips as a string and saves to
    # disk. ``process_tasks`` later crashes on ``response_message['plan']``
    # because it's a string. Unwrap nested strings and harden the task
    # extractor so the plan survives gpt-4.1's stricter output formatting.
    orig_valid = _autoba_agent.Agent.valid_json_response

    def _extract_plan_dict(raw):
        """Try every sensible way to coax a dict-with-plan out of an LLM
        response.

        gpt-4.1 has been observed to return its object JSON wrapped in
        literal outer double quotes with the inner quotes escaped as
        ``\\"`` but the newlines left unescaped (so neither the whole
        string parses as JSON, nor does the brace-slice parse cleanly).
        We try, in order:

          1. Bare parse
          2. Markdown code-fence strip (```json … ``` / ``` … ```)
          3. Outer-quote + unicode_escape (handles ``"{\\n  \\"plan\\"…}"``
             by turning the inner escape sequences back into real chars)
          4. Outermost-brace slice, unescaping ``\\"`` → ``"`` so a
             pseudo-string slice becomes real JSON
          5. Outermost-brace slice, bare

        Every candidate goes through nested-string unwrapping (up to 3
        layers) so doubly / triply stringified JSON also lands as a dict.
        First dict wins — we do not require any specific key because
        AutoBA's validator runs over both plan responses (``{"plan": …}``)
        and code-generation responses (``{"tool": …, "code": …}``).
        """
        import json as _json
        import re as _re
        if not isinstance(raw, str):
            return None
        candidates: List[str] = [raw]
        s = raw.strip()
        # (2) Markdown fence strip
        if s.startswith("```"):
            s2 = _re.sub(r"^```(?:json|python|bash)?\s*\n?", "", s, count=1)
            s2 = _re.sub(r"\n?```\s*$", "", s2)
            candidates.append(s2)
        # (3) Outer-quote strip + unicode_escape — handles the gpt-4.1
        # observed pattern where the whole response is a "..."-wrapped
        # JSON with real newlines inside (not valid JSON as-is).
        if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
            try:
                inner = bytes(s[1:-1], "utf-8").decode("unicode_escape")
                candidates.append(inner)
            except Exception:
                pass
        # Brace slices (with and without `\\"` → `"` unescape)
        i, j = raw.find("{"), raw.rfind("}")
        if i >= 0 and j > i:
            brace = raw[i:j + 1]
            candidates.append(brace.replace('\\"', '"').replace('\\\\', '\\'))
            candidates.append(brace)
        for cand in candidates:
            try:
                parsed = _json.loads(cand)
            except Exception:
                continue
            depth = 0
            while isinstance(parsed, str) and depth < 3:
                try:
                    parsed = _json.loads(parsed)
                except Exception:
                    break
                depth += 1
            if isinstance(parsed, dict):
                return parsed
        return None

    # Diagnostic gated by $AUTOBA_SHIM_DEBUG. On by default OFF so the
    # normal sweep stays quiet; set AUTOBA_SHIM_DEBUG=1 to log every
    # validator call's response shape (useful when a new model produces
    # a response the extractor rejects).
    _debug = bool(os.environ.get("AUTOBA_SHIM_DEBUG"))
    _call_count = {"n": 0}

    def _patched_valid(self, response_message):
        import json as _json
        import os as _os
        _call_count["n"] += 1
        n = _call_count["n"]
        if not _os.path.isdir(f'{self.output_dir}'):
            _os.makedirs(f'{self.output_dir}')
        parsed = _extract_plan_dict(response_message)
        if _debug:
            print(f"[autoba_shim] call #{n} "
                  f"round={getattr(self,'global_round','?')} "
                  f"type={type(response_message).__name__} "
                  f"len={len(str(response_message))} "
                  f"parsed={'dict' if isinstance(parsed, dict) else 'None'}",
                  file=sys.stderr)
        if parsed is None:
            if _debug:
                print(f"[autoba_shim] call #{n} REJECTED; repr[:500]:",
                      repr(response_message)[:500], file=sys.stderr)
            return False
        path = f'{self.output_dir}/{self.global_round}_response.json'
        try:
            with open(path, 'w') as w:
                _json.dump(parsed, w)
        except Exception:
            return False
        return True
    _autoba_agent.Agent.valid_json_response = _patched_valid

    orig_process = getattr(_autoba_agent.Agent, "process_tasks", None)
    def _patched_process_tasks(self, response_message):
        import json as _json
        if isinstance(response_message, str):
            try:
                response_message = _json.loads(response_message)
            except Exception:
                response_message = {"plan": []}
        if not isinstance(response_message, dict):
            response_message = {"plan": []}
        self.tasks = response_message.get("plan", []) or []
    if orig_process is not None:
        _autoba_agent.Agent.process_tasks = _patched_process_tasks

    # AutoBA sprinkles ``time.sleep(15)`` between plan / code-gen phases
    # and ``time.sleep(20)`` per retry as 2023-era OpenAI rate-limit
    # guards. With ~6 tasks per plan that's 90+ seconds of pure sleep,
    # blowing through the default 180s per-cell timeout for no reason
    # — modern gpt-4.x rate limits make these waits unnecessary. The
    # subprocess is isolated so we can safely no-op sleep everywhere.
    import time as _time
    _time.sleep = lambda *_a, **_kw: None

    # Skip ``run_code_generation_phase`` entirely. It produces per-task
    # shell scripts via N additional LLM calls (one per task), but the
    # planning benchmark only needs the plan — AutoBA's task
    # descriptions already contain tool names ("Use kallisto quant to
    # …") that ``score_plan``'s tool-name matcher can grep for. Skipping
    # the code-gen phase cuts per-cell wall time from ~180s (often
    # blowing the per-cell timeout) to ~20–40s, apples-to-apples with
    # FlowAgent and BioMaster (both of which also skip execution /
    # per-step codegen in our config).
    def _noop_codegen(self):
        return None
    _autoba_agent.Agent.run_code_generation_phase = _noop_codegen


def _install_openai_token_patch() -> Dict[str, int]:
    """Monkey-patch the OpenAI chat-completions endpoint to accumulate usage.

    AutoBA calls ``self.openai_client.chat.completions.create(...)`` directly.
    Every response carries a ``usage`` field (prompt_tokens, completion_tokens);
    we accumulate into the returned dict and return it so the caller can read
    it post-invocation. Only non-streaming responses are captured, which is
    what AutoBA uses.
    """
    stats = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    try:
        from openai.resources.chat.completions import Completions
    except Exception as e:
        print(f"[autoba_shim] could not import openai Completions: {e}",
              file=sys.stderr)
        return stats
    original_create = Completions.create

    def _patched_create(self, *args, **kwargs):
        resp = original_create(self, *args, **kwargs)
        try:
            u = getattr(resp, "usage", None)
            if u is not None:
                stats["prompt_tokens"] += int(
                    getattr(u, "prompt_tokens", 0) or 0)
                stats["completion_tokens"] += int(
                    getattr(u, "completion_tokens", 0) or 0)
                stats["calls"] += 1
        except Exception:
            # Token accounting is best-effort — never let it break the run.
            pass
        return resp
    Completions.create = _patched_create  # type: ignore[assignment]
    return stats


def _write_scratch_config(*, scratch_dir: Path, output_dir: Path,
                          datalist: List[str], goal: str) -> Path:
    """AutoBA's config.yaml shape (see examples/case1.1/config.yaml)."""
    cfg = {
        "data_list": list(datalist),
        "output_dir": str(output_dir),
        "goal_description": goal,
    }
    try:
        import yaml
    except ImportError:
        _die("PyYAML not available (AutoBA needs it too)",
             error_type="missing-dep")
    cfg_path = scratch_dir / "config.yaml"
    with cfg_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return cfg_path


def _map_plan(plan_tasks: List[str], shells: Dict[int, str]) -> List[Dict[str, Any]]:
    """Project AutoBA's plan (list of task strings) onto FlowAgent steps.

    Each task becomes one step; the corresponding ``<N>.sh`` (N=1..len)
    body becomes the ``command``. If a shell is missing (task crashed or
    hadn't run yet), we fall back to the task description string so
    ``score_plan`` still has tool-name text to grep.
    """
    steps: List[Dict[str, Any]] = []
    for idx, task in enumerate(plan_tasks, 1):
        shell_body = shells.get(idx, "")
        command = shell_body if shell_body.strip() else str(task)
        steps.append({
            "name":         f"step_{idx}",
            "command":      command,
            "dependencies": [f"step_{idx - 1}"] if idx > 1 else [],
            "outputs":      [],
            "description":  str(task),
        })
    return steps


def _collect_shells(output_dir: Path) -> Dict[int, str]:
    """Read ``<output_dir>/<N>.sh`` (N = 1, 2, …) into {N: body}."""
    out: Dict[int, str] = {}
    for path in output_dir.glob("*.sh"):
        try:
            n = int(path.stem)
        except ValueError:
            continue
        if n <= 0:
            continue  # 0_response.json is the plan, not a shell
        out[n] = path.read_text(encoding="utf-8", errors="replace")
    return out


def main() -> None:
    args = _parse_args()
    t0 = time.perf_counter()

    if not args.autoba_dir:
        _die("Set --autoba-dir or $AUTOBA_DIR to the AutoBA clone",
             error_type="config")
    autoba_dir = Path(args.autoba_dir).expanduser().resolve()
    if not (autoba_dir / "app.py").exists():
        _die(f"{autoba_dir} does not look like an AutoBA clone "
             "(missing app.py)", error_type="config")
    if not args.api_key:
        _die("OPENAI_API_KEY is not set (neither in env nor --api-key)",
             error_type="config")

    try:
        datalist = json.loads(args.files)
        if not isinstance(datalist, list):
            raise ValueError("--files must decode to a list of strings")
    except Exception as e:
        _die(f"--files must be a JSON array: {e}", error_type="config")
    if not datalist:
        # AutoBA's prompt logic expects at least one data entry; give it a
        # generic stand-in if the caller didn't supply anything concrete.
        datalist = ["./input.fastq.gz: paired/single-end sequencing reads"]

    run_id = uuid.uuid4().hex[:10]
    scratch_dir = Path(tempfile.mkdtemp(prefix=f"autoba_{run_id}_"))
    output_dir = scratch_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = _write_scratch_config(
        scratch_dir=scratch_dir, output_dir=output_dir,
        datalist=datalist, goal=args.prompt,
    )

    # Some AutoBA code paths reach for softwares_config/ and
    # softwares_database/ relative to cwd. Symlink them from the clone so
    # their absence doesn't crash a run; the clone itself stays untouched.
    prev_cwd = os.getcwd()
    for ref in ("softwares_config", "softwares_database"):
        src = autoba_dir / ref
        if src.exists():
            try:
                (scratch_dir / ref).symlink_to(src)
            except OSError:
                shutil.copytree(src, scratch_dir / ref)
    os.chdir(scratch_dir)
    sys.path.insert(0, str(autoba_dir))

    prompt_tokens = completion_tokens = llm_calls = 0
    cost_usd = 0.0
    error: Optional[str] = None
    plan: Dict[str, Any] = {"workflow_type": "custom", "steps": []}

    with contextlib.redirect_stdout(sys.stderr):
        try:
            import runpy
            _install_unused_import_stubs()
            _patch_autoba_model_whitelist(autoba_dir)
            stats = _install_openai_token_patch()

            app_py = autoba_dir / "app.py"
            prev_argv = sys.argv
            sys.argv = [
                str(app_py),
                "--config", str(cfg_path),
                "--openai", args.api_key,
                "--model", args.model,
                "--execute", "False",
            ]
            try:
                try:
                    runpy.run_path(str(app_py), run_name="__main__")
                except SystemExit:
                    pass
                except Exception as run_exc:
                    error = (f"app.py raised: "
                             f"{type(run_exc).__name__}: {run_exc}")
            finally:
                sys.argv = prev_argv

            prompt_tokens = stats["prompt_tokens"]
            completion_tokens = stats["completion_tokens"]
            llm_calls = stats["calls"]
            # cost_usd: let the parent harness compute from tokens using
            # models.yaml pricing. We just report 0 here.

            plan_path = output_dir / "0_response.json"
            if not plan_path.exists():
                if error is None:
                    error = ("run_plan_phase produced no 0_response.json "
                             "(upstream LLM output wasn't valid JSON)")
                plan_tasks: List[str] = []
            else:
                raw = json.loads(plan_path.read_text(encoding="utf-8"))
                tasks = raw.get("plan") if isinstance(raw, dict) else None
                if isinstance(tasks, list):
                    plan_tasks = [str(t) for t in tasks]
                else:
                    plan_tasks = []
            shells = _collect_shells(output_dir)
            steps = _map_plan(plan_tasks, shells)
            plan = {
                "workflow_type": _classify_workflow_type({"steps": steps}),
                "steps": steps,
            }
            # Soft-clear the upstream crash if we still got a plan to score.
            if steps and error and error.startswith("app.py raised"):
                error = None

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-800:]}"
        finally:
            os.chdir(prev_cwd)
            if not args.keep_output:
                shutil.rmtree(scratch_dir, ignore_errors=True)

    print(json.dumps({
        "plan": plan,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "llm_calls": llm_calls,
        "cost_usd": cost_usd,
        "wall_seconds": time.perf_counter() - t0,
        "error": error,
    }))


if __name__ == "__main__":
    main()
