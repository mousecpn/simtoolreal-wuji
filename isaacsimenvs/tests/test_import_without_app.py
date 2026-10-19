"""Check task registration without booting Isaac Sim.

Run with .venv_isaacsim/bin/python isaacsimenvs/tests/test_import_without_app.py.
"""

from pathlib import Path
import subprocess
import sys
import unittest


class TestImportWithoutApp(unittest.TestCase):
    def test_registration_without_simulation_imports(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
import torch
import isaaclab
import isaacsimenvs
import gymnasium as gym

spec = gym.spec("Isaacsimenvs-SimToolReal-Direct-v0")
assert spec.entry_point == "isaacsimenvs.tasks.simtoolreal.simtoolreal_env:SimToolRealEnv"
assert spec.kwargs["env_cfg_entry_point"] == "isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg:SimToolRealEnvCfg"
assert "isaacsimenvs.tasks.simtoolreal.simtoolreal_env" not in sys.modules
assert "isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg" not in sys.modules
for task, package, cls in (
    ("TianjiWuji", "tianji_wuji", "TianjiWujiEnv"),
    ("TianjiWujiDual", "tianji_wuji_dual", "TianjiWujiDualEnv"),
):
    registered = gym.spec(f"Isaacsimenvs-SimToolReal-{task}-Direct-v0")
    assert registered.entry_point == f"isaacsimenvs.tasks.{package}.env:{cls}"
    assert f"isaacsimenvs.tasks.{package}.env" not in sys.modules
    assert f"isaacsimenvs.tasks.{package}.config" not in sys.modules
assert "isaaclab.envs" not in sys.modules
""",
            ],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()