"""Opt-in registry entry; delegates training unchanged to vanilla train.main."""
from pathlib import Path
import gymnasium as gym


def main():
    root=Path(__file__).resolve().parents[1]
    gym.register(
        id='Isaacsimenvs-SingleHammerBenchmark-Direct-v0',
        entry_point='isaacsimenvs.tasks.simtoolreal.simtoolreal_env:SimToolRealEnv',
        disable_env_checker=True,
        kwargs={
            'env_cfg_entry_point':'supermimic_stl.benchmark:make_env_cfg',
            'rl_games_sapg_cfg_entry_point':str(root/'isaacsimenvs/cfg/train/SimToolRealSAPG.yaml'),
        })
    from isaacsimenvs.train import main as vanilla_main
    vanilla_main()

if __name__=='__main__': main()
