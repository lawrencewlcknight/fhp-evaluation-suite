"""Opt-in native repo gates, isolated to prevent UCV/VR namespace collisions."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("family,variable", [("ucv", "FHP_UCV_REPO"),
                                            ("vr_deep", "FHP_VR_REPO"),
                                            ("sd_cfr", "FHP_SD_REPO")])
def test_native_checkpoint_variants(tmp_path, family, variable):
    repo = os.environ.get(variable)
    if not repo:
        pytest.skip(f"Set {variable} to enable native checkpoint integration tests")
    suite = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(suite))
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("native_adapter_probe.py")),
                             family, repo, str(tmp_path)], capture_output=True, text=True,
                            env=env, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"passed": true' in result.stdout
