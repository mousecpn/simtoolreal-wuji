"""Registration only: simulator imports remain lazy until AppLauncher starts."""
from pathlib import Path
import gymnasium as gym

_CFG = Path(__file__).resolve().parents[2] / "cfg" / "train"
gym.register(
    id="Isaacsimenvs-SimToolReal-TianjiWujiDual-Direct-v0",
    entry_point="isaacsimenvs.tasks.tianji_wuji_dual.env:TianjiWujiDualEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.tianji_wuji_dual.config:TianjiWujiDualEnvCfg",
        "rl_games_cfg_entry_point": str(_CFG / "TianjiWujiDualPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG / "TianjiWujiDualSAPG.yaml"),
    },
)
