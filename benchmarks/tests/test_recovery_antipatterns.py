"""Unit tests for ``WorkflowManager._is_recovery_antipattern``.

The reviewer flagged that the previous antipattern detector caught
``rm`` / ``echo`` shapes but advertised broader coverage in its
docstring. The fix expands the rejection set to:

  - bare ``true`` / ``:`` / ``exit 0`` / ``test 0`` no-ops
  - trailing ``|| true`` / ``|| continue`` / ``|| :`` failure
    suppression
  - leading ``set +e`` swallowing errors
  - "recovery" that drops the original tool family and leaves only
    shell builtins (no real tool invocation)

These tests assert each shape rejects, and that legitimate recoveries
still pass through.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))


def _stub_optional_modules() -> None:
    """Stub heavy optional deps so we can import workflow_manager.

    The recovery antipattern check is pure-Python and uses ``re`` only,
    but the module that owns it pulls in (transitively) ``openai``,
    ``anthropic``, etc. via ``flowagent.core.llm``. Test environments
    that don't have those installed should still be able to exercise
    the static method.
    """
    for name in ("openai", "anthropic", "google", "google.generativeai",
                 "tiktoken"):
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        # ``from openai import AsyncOpenAI`` etc. — supply attribute hooks.
        mod.AsyncOpenAI = type("AsyncOpenAI", (), {})            # type: ignore[attr-defined]
        mod.AsyncAnthropic = type("AsyncAnthropic", (), {})      # type: ignore[attr-defined]
        mod.GenerativeModel = type("GenerativeModel", (), {})    # type: ignore[attr-defined]
        sys.modules[name] = mod


_stub_optional_modules()

from flowagent.core.workflow_manager import WorkflowManager  # noqa: E402


_REJECT = WorkflowManager._is_recovery_antipattern


class TestBareNoOps:
    def test_bare_true(self):
        assert _REJECT("true") is not None

    def test_bare_colon(self):
        assert _REJECT(":") is not None

    def test_bare_exit_zero(self):
        assert _REJECT("exit 0") is not None

    def test_padded_true(self):
        assert _REJECT("   true   ") is not None

    def test_test_zero(self):
        assert _REJECT("test 0") is not None

    def test_absolute_path_true(self):
        assert _REJECT("/bin/true") is not None
        assert _REJECT("/usr/bin/true") is not None


class TestTrailingFailureSuppression:
    def test_or_true(self):
        assert _REJECT("samtools sort in.bam || true") is not None

    def test_or_continue(self):
        assert _REJECT("kallisto quant ... || continue") is not None

    def test_or_colon(self):
        assert _REJECT("multiqc out/ || :") is not None

    def test_or_exit_0(self):
        assert _REJECT("bwa mem ref reads.fq.gz || exit 0") is not None


class TestLeadingSetPlusE:
    def test_set_plus_e_then_work(self):
        assert _REJECT("set +e; samtools sort in.bam") is not None

    def test_set_plus_e_with_export(self):
        assert _REJECT("set +e\nbwa mem ref reads.fq.gz") is not None


class TestEchoOnly:
    def test_bare_echo_message(self):
        assert _REJECT("echo 'failed: skipping'") is not None

    def test_echo_then_exit(self):
        assert _REJECT("echo 'fail'; exit 0") is not None


class TestRmOnly:
    def test_rm_then_echo_please_redownload(self):
        cmd = "rm -f input.tar.gz && echo 'please redownload manually'"
        assert _REJECT(cmd) is not None


class TestToolFamilyCheck:
    def test_real_tool_replaced_by_only_builtins_rejects(self):
        original = "samtools sort -o sorted.bam in.bam"
        fix = "mkdir -p out && touch out/sorted.bam"
        # ``samtools`` is dropped and only mkdir/touch remain — cheat.
        assert _REJECT(fix, original_command=original) is not None

    def test_real_tool_kept_passes(self):
        original = "samtools sort -o sorted.bam in.bam"
        fix = "mkdir -p out && samtools sort -@4 -o out/sorted.bam in.bam"
        assert _REJECT(fix, original_command=original) is None


class TestLegitimateRecoveriesPass:
    def test_mkdir_then_real_command(self):
        cmd = "mkdir -p quant/ && kallisto quant -i tx.idx -o quant/ r.fq.gz"
        assert _REJECT(cmd) is None

    def test_explicit_input_check_then_real_command(self):
        cmd = "[ -s reads.fq.gz ] || exit 1; kallisto quant -i tx.idx -o quant/ reads.fq.gz"
        assert _REJECT(cmd) is None

    def test_swap_to_curl_from_wget(self):
        cmd = "curl -fsSL -o reads.fq.gz http://example.com/r.fq.gz"
        original = "wget -O reads.fq.gz http://example.com/r.fq.gz"
        assert _REJECT(cmd, original_command=original) is None

    def test_empty_command_rejects(self):
        assert _REJECT("") is not None
        assert _REJECT("   ") is not None
