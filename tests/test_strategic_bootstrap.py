"""Regress the Exp16 cloud bootstrap without apt, Python installs or paid jobs."""

import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "gcp/run_exp6_ucv_exp16_strategic_audit.sh"
STAGES = ("system_packages", "bootstrap_venv", "uv_install", "python_availability",
          "managed_python", "evaluation_venv", "runtime_version", "torch_install",
          "evaluation_dependencies", "dependency_check")


def body():
    return SCRIPT.read_text().split("<<'BOOTSTRAP'\n", 1)[1].split("\nBOOTSTRAP\n", 1)[0]


def catalogue_check():
    return body().split("<<'PYTHON_CATALOG_CHECK'\n", 1)[1].split("\nPYTHON_CATALOG_CHECK\n", 1)[0]


def checked_pins():
    python = re.search(r"^FHP_AUDIT_PYTHON_VERSION=(\S+)$", body(), re.M).group(1)
    uv = re.search(r"^FHP_AUDIT_UV_VERSION=(\S+)$", body(), re.M).group(1)
    return python, uv


def run_check(tmp_path, rows, request):
    path = tmp_path / "downloads.json"
    path.write_text(json.dumps(rows))
    return subprocess.run([sys.executable, "-I", "-", str(path), request],
                          input=catalogue_check(), capture_output=True, text=True, timeout=15)


def test_runtime_pins_are_consistent_with_working_evaluator():
    assert checked_pins() == ("3.11.13", "0.8.22")
    for command in ('python install "$FHP_AUDIT_PYTHON_VERSION"',
                    'venv --python "$FHP_AUDIT_PYTHON_VERSION" --managed-python --no-python-downloads'):
        assert command in body()
    assert body().index("bootstrap_stage python_availability") < body().index("bootstrap_stage managed_python")
    assert body().index("bootstrap_stage runtime_version") < body().index("bootstrap_stage torch_install")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


@pytest.mark.parametrize("case", ["missing", "wrong_arch", "wrong_python", "installed_only", "duplicate", "malformed"])
def test_catalogue_check_rejects_unavailable_or_wrong_build(tmp_path, case):
    request = "cpython-3.11.13-linux-x86_64-gnu"
    row = dict(key=request, url="https://example.invalid/cpython.tar.gz")
    rows = [row]
    if case == "missing": rows = []
    elif case == "wrong_arch": row["key"] = "cpython-3.11.13-macos-aarch64-none"
    elif case == "wrong_python": row["key"] = "cpython-3.11.16-linux-x86_64-gnu"
    elif case == "installed_only": row["url"] = None
    elif case == "duplicate": rows.append(row)
    elif case == "malformed": rows = {"key": request}
    result = run_check(tmp_path, rows, request)
    assert result.returncode != 0
    assert "align the Python and uv pins" in result.stderr


def test_catalogue_check_accepts_exact_download(tmp_path):
    request = "cpython-3.11.13-linux-x86_64-gnu"
    result = run_check(tmp_path, [dict(key=request, url="https://example.invalid/cpython.tar.gz")], request)
    assert result.returncode == 0, result.stderr
    assert request in result.stdout


@pytest.mark.parametrize("failure_stage", [None, *STAGES])
def test_submitted_bootstrap_stage_order_and_fail_fast(tmp_path, failure_stage):
    work = tmp_path / "work"
    (work / "output").mkdir(parents=True)
    mock = """
mock_command() {
  printf 'MOCK command=%s stage=%s\n' "$*" "$FHP_BOOTSTRAP_STAGE"
  if [[ "$FHP_BOOTSTRAP_STAGE" == "${FHP_TEST_FAIL_STAGE:-}" ]]; then return 37; fi
}
"""
    for name in ("apt-get", str(work / "bootstrap-venv/bin/pip"),
                 str(work / "venv/bin/pip"), str(work / "venv/bin/python")):
        mock += f'function {shlex.quote(name)}() {{ mock_command "{name}" "$@"; }}\n'
    mock += f"""
function /usr/bin/python3() {{
  if [[ "$2" == "-" ]]; then {shlex.quote(sys.executable)} "$@";
  else mock_command /usr/bin/python3 "$@"; fi
}}
function {shlex.quote(str(work / 'bootstrap-venv/bin/uv'))}() {{
  mock_command uv "$@" >&2 || return $?
  if [[ "$1 $2" == "python list" ]]; then
    printf '[{{"key":"%s","url":"https://example.invalid/cpython.tar.gz"}}]\n' "$3"
  fi
}}
"""
    result = subprocess.run(["/bin/bash", "-Eeuo", "pipefail", "-c", mock + body()],
                            env=dict(os.environ, WORK=str(work), FHP_TEST_FAIL_STAGE=failure_stage or ""),
                            capture_output=True, text=True, timeout=15)
    if failure_stage:
        assert result.returncode == 37, result.stdout + result.stderr
        assert f"ERROR: BOOTSTRAP stage={failure_stage}" in result.stderr
        for later in STAGES[STAGES.index(failure_stage) + 1:]:
            assert f"BOOTSTRAP stage={later}\n" not in result.stdout
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Verified managed Python download: cpython-3.11.13-linux-x86_64-gnu" in result.stdout
        assert "uv==0.8.22" in result.stdout
        assert "python install 3.11.13" in result.stderr
        assert "venv --python 3.11.13" in result.stderr
        assert "torch==2.7.0+cpu" in result.stdout
        assert "BOOTSTRAP stage=complete" in result.stdout


def test_real_pinned_uv_catalogue_regresses_original_failure(tmp_path):
    """Optional release gate: real pinned executable, offline, Linux catalogue on any host."""
    uv = os.environ.get("FHP_TEST_UV")
    if not uv:
        pytest.skip("Set FHP_TEST_UV to an existing uv 0.8.22 executable for the offline release gate")
    python_version, uv_version = checked_pins()
    result = subprocess.run([uv, "--version"], capture_output=True, text=True, check=True, timeout=15)
    assert result.stdout.split()[:2] == ["uv", uv_version]
    for version, expected in ((python_version, 0), ("3.11.16", 1)):
        request = f"cpython-{version}-linux-x86_64-gnu"
        result = subprocess.run([uv, "python", "list", request, "--all-versions", "--all-platforms",
                                 "--all-arches", "--only-downloads", "--output-format", "json",
                                 "--offline", "--no-cache", "--no-config"],
                                capture_output=True, text=True, check=True, timeout=30)
        checked = run_check(tmp_path, json.loads(result.stdout), request)
        assert checked.returncode == expected, checked.stdout + checked.stderr
