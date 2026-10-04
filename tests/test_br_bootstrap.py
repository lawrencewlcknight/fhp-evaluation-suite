"""Exercise the submitted shell with minimal environments, without cloud calls.

Real venv checks use this test process's Python (not a claim of Debian/GCP
acceptance). Package-install flow tests mock external commands, not shell logic.
"""

import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "gcp/run_exp4_ucv_exp9_br_pilot.sh"
FINAL = ROOT / "gcp/finalize_exp4_ucv_exp9_br_pilot.sh"


def path_setup(script):
    return next(line for line in script.read_text().splitlines() if line.startswith("export PATH="))


def bootstrap_body():
    return MAIN.read_text().split("<<'FHP_BOOTSTRAP'\n", 1)[1].split("\nFHP_BOOTSTRAP\n", 1)[0]


@pytest.mark.parametrize("script", [MAIN, FINAL])
@pytest.mark.parametrize("initial_path", [None, "", "/custom/cloud-sdk/bin"])
def test_path_is_exported_to_children_with_system_and_existing_entries(script, initial_path):
    environment = {} if initial_path is None else {"PATH": initial_path}
    # A nested shell reproduces the environment boundary in the timed bootstrap.
    probe = "import json, os; print(json.dumps(os.environ['PATH']))"
    child = f"{shlex.quote(sys.executable)} -I -c {shlex.quote(probe)}"
    result = subprocess.run(["/bin/bash", "-Eeuo", "pipefail", "-c",
                             path_setup(script) + "\n/bin/bash -c " + shlex.quote(child)],
                            env=environment, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    entries = json.loads(result.stdout).split(":")
    assert entries[:6] == ["/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin"]
    assert "" not in entries
    if initial_path:
        assert initial_path in entries


@pytest.mark.parametrize("initial_path", [None, ""])
def test_real_venv_creation_with_no_inherited_path_or_usable_pythonhome(tmp_path, initial_path):
    body = bootstrap_body()
    diagnostic = next(line for line in body.splitlines() if line.startswith("/usr/bin/python3 -I -c "))
    creation = next(line for line in body.splitlines() if line.startswith("/usr/bin/python3 -I -m venv "))
    # Keep the production invocation/options, substituting only platform/path.
    venv = tmp_path / "bootstrap venv"
    script = (path_setup(MAIN) + "\n" + diagnostic + "\n" + creation).replace(
        "/usr/bin/python3", shlex.quote(sys.executable)).replace(
        "/workspace/fhp-br-exp9/bootstrap-venv", shlex.quote(str(venv)))
    environment = {"PYTHONHOME": "/nonexistent-bootstrap-test", "PYTHONPATH": "/nonexistent-bootstrap-test"}
    if initial_path is not None:
        environment["PATH"] = initial_path
    result = subprocess.run(["/bin/bash", "-Eeuo", "pipefail", "-c", script],
                            env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    diagnostic_data = json.loads(result.stdout)
    assert Path(diagnostic_data["base_executable"]).is_absolute()
    assert (venv / "bin/python").is_file()
    check = subprocess.run([str(venv / "bin/python"), "-I", "-c",
                            "import sys; assert sys.prefix != sys.base_prefix"],
                           env=environment, capture_output=True, text=True, timeout=15)
    assert check.returncode == 0, check.stderr


STAGES = ["system_packages", "python_diagnostics", "bootstrap_venv", "uv_install",
          "managed_python", "evaluation_venv", "torch_install", "evaluation_dependencies",
          "dependency_check"]


@pytest.mark.parametrize("failure_stage", [None, *STAGES])
def test_setup_stage_logging_and_fail_fast_without_installing_packages(failure_stage):
    # Bash permits absolute-path function names. Mock these exact commands so the
    # submitted bootstrap body runs unchanged, with no apt/pip/uv installs.
    mock = """
mock_install_command() {
  printf 'MOCK command=%s stage=%s\n' "$*" "$FHP_BOOTSTRAP_STAGE"
  if [[ "$FHP_BOOTSTRAP_STAGE" == "${FHP_TEST_FAIL_STAGE:-}" ]]; then return 37; fi
}
"""
    for command in ("apt-get", "/usr/bin/python3", "/workspace/fhp-br-exp9/bootstrap-venv/bin/pip",
                    "/workspace/fhp-br-exp9/bootstrap-venv/bin/uv", "/workspace/fhp-br-exp9/venv/bin/pip"):
        mock += f'function {command}() {{ mock_install_command "{command}" "$@"; }}\n'
    result = subprocess.run(["/bin/bash", "-Eeuo", "pipefail", "-c",
                             path_setup(MAIN) + "\n" + mock + bootstrap_body()],
                            env={"FHP_TEST_FAIL_STAGE": failure_stage or ""},
                            capture_output=True, text=True, timeout=15)
    if failure_stage:
        assert result.returncode == 37, result.stdout + result.stderr
        assert f"ERROR: BOOTSTRAP stage={failure_stage} line=" in result.stderr
        assert "exit_code=37" in result.stderr
        assert "BOOTSTRAP stage=complete" not in result.stdout
        for later in STAGES[STAGES.index(failure_stage) + 1:]:
            assert f"BOOTSTRAP stage={later}\n" not in result.stdout
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        for stage in [*STAGES, "complete"]:
            assert f"BOOTSTRAP stage={stage}\n" in result.stdout
        assert "python3 -I -m venv" in result.stdout
        assert "uv==0.8.22" in result.stdout
        assert "python install 3.11.13" in result.stdout
        assert "venv --python 3.11.13" in result.stdout
        assert "torch==2.7.0+cpu" in result.stdout
