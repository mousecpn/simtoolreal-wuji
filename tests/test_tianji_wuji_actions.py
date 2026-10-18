"""CPU checks for canonical policy actions with deliberately permuted joints."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


SPEC = importlib.util.spec_from_file_location(
    "tianji_wuji_actions",
    Path(__file__).resolve().parents[1] / "isaacsimenvs/tasks/tianji_wuji/actions.py",
)
actions_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(actions_module)


class ActionTests(unittest.TestCase):
    def make_env(self, delay=False):
        # Lab order deliberately interleaves fingers and arm joints.
        permutation = torch.randperm(27, generator=torch.Generator().manual_seed(17))
        arm = torch.nonzero(permutation < 7).flatten()
        hand = torch.nonzero(permutation >= 7).flatten()
        lower = -torch.arange(1, 28, dtype=torch.float32) / 20
        upper = torch.arange(1, 28, dtype=torch.float32) / 10
        return SimpleNamespace(
            num_envs=2, device="cpu", step_dt=.02,
            cfg=SimpleNamespace(
                domain_randomization=SimpleNamespace(use_action_delay=delay, action_delay_max=3),
                action=SimpleNamespace(dof_speed_scale=1.5, arm_moving_average=.3, hand_moving_average=.4),
            ),
            _perm_canon_to_lab=permutation,
            _arm_joint_ids=arm, _hand_joint_ids=hand,
            _arm_lower=lower[arm], _arm_upper=upper[arm],
            _hand_lower=lower[hand], _hand_upper=upper[hand],
            _prev_targets=torch.zeros(2, 27), _cur_targets=torch.zeros(2, 27),
            _action_queue=torch.zeros(2, 3, 27), episode_length_buf=torch.zeros(2, dtype=torch.long),
            _successes=torch.zeros(2, dtype=torch.long),
        )

    def test_each_canonical_action_reaches_named_joint(self):
        env = self.make_env()
        values = torch.linspace(-1.4, 1.4, 27).repeat(2, 1)
        values[1] = -values[1]
        actions_module.apply_action_pipeline(env, values)
        for lab_id, canonical_id in enumerate(env._perm_canon_to_lab.tolist()):
            action = values[:, canonical_id].clamp(-1, 1)
            if canonical_id < 7:
                expected = .3 * 1.5 * .02 * action
            else:
                lower, upper = -(lab_id + 1) / 20, (lab_id + 1) / 10
                expected = .4 * (lower + .5 * (action + 1) * (upper - lower))
            torch.testing.assert_close(env._cur_targets[:, lab_id], expected)
        torch.testing.assert_close(env._prev_targets, env._cur_targets)

    def test_arm_saturation_and_hand_target_limits(self):
        env = self.make_env()
        env._prev_targets[:, env._arm_joint_ids] = env._arm_upper
        actions_module.apply_action_pipeline(env, torch.ones(2, 27))
        torch.testing.assert_close(env._cur_targets[:, env._arm_joint_ids], env._arm_upper.expand(2, -1))
        self.assertTrue((env._cur_targets[:, env._hand_joint_ids] <= env._hand_upper).all())
        self.assertTrue((env._cur_targets[:, env._hand_joint_ids] >= env._hand_lower).all())

    def test_episode_start_flushes_delayed_actions(self):
        env, reference = self.make_env(delay=True), self.make_env()
        env._action_queue.fill_(123)
        values = torch.linspace(-1, 1, 27).repeat(2, 1)
        actions_module.apply_action_pipeline(env, values)
        actions_module.apply_action_pipeline(reference, values)
        torch.testing.assert_close(env._cur_targets, reference._cur_targets)

    def test_rejects_incompatible_action_shape(self):
        with self.assertRaisesRegex(ValueError, "27"):
            actions_module.apply_action_pipeline(self.make_env(), torch.zeros(2, 29))


if __name__ == "__main__":
    unittest.main()
