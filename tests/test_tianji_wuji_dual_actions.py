"""Dual policy actions retain left/right identity under shuffled articulation order."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


SPEC = importlib.util.spec_from_file_location(
    "tianji_wuji_dual_actions",
    Path(__file__).resolve().parents[1] / "isaacsimenvs/tasks/tianji_wuji_dual/actions.py",
)
actions_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(actions_module)


class DualActionTests(unittest.TestCase):
    def make_env(self, delay=False):
        permutation = torch.randperm(54, generator=torch.Generator().manual_seed(17))
        is_arm = (permutation < 7) | ((permutation >= 27) & (permutation < 34))
        arm = torch.nonzero(is_arm).flatten()
        hand = torch.nonzero(~is_arm).flatten()
        lower = -torch.arange(1, 55, dtype=torch.float32) / 20
        upper = torch.arange(1, 55, dtype=torch.float32) / 10
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
            _prev_targets=torch.zeros(2, 54), _cur_targets=torch.zeros(2, 54),
            _action_queue=torch.zeros(2, 3, 54), episode_length_buf=torch.zeros(2, dtype=torch.long),
            _successes=torch.zeros(2, dtype=torch.long),
        )

    def test_each_canonical_action_reaches_named_joint(self):
        env = self.make_env()
        values = torch.linspace(-1.4, 1.4, 54).repeat(2, 1)
        values[1] = -values[1]
        actions_module.apply_action_pipeline(env, values)
        for lab_id, canonical_id in enumerate(env._perm_canon_to_lab.tolist()):
            action = values[:, canonical_id].clamp(-1, 1)
            if canonical_id < 7 or 27 <= canonical_id < 34:
                expected = .3 * 1.5 * .02 * action
            else:
                lower, upper = -(lab_id + 1) / 20, (lab_id + 1) / 10
                expected = .4 * (lower + .5 * (action + 1) * (upper - lower))
            torch.testing.assert_close(env._cur_targets[:, lab_id], expected)
        torch.testing.assert_close(env._prev_targets, env._cur_targets)

    def test_left_and_right_actions_are_independent(self):
        baseline, changed = self.make_env(), self.make_env()
        zeros = torch.zeros(2, 54)
        values = zeros.clone()
        values[0, :27] = 1  # Only left side moves in environment 0.
        values[1, 27:] = -1  # Only right side moves in environment 1.
        actions_module.apply_action_pipeline(baseline, zeros)
        actions_module.apply_action_pipeline(changed, values)
        left = changed._perm_canon_to_lab < 27
        for row, commanded in ((0, left), (1, ~left)):
            torch.testing.assert_close(changed._cur_targets[row, ~commanded], baseline._cur_targets[row, ~commanded])
            self.assertTrue(torch.all(changed._cur_targets[row, commanded] != baseline._cur_targets[row, commanded]))

    def test_both_arm_saturation_and_hand_target_limits(self):
        env = self.make_env()
        env._prev_targets[:, env._arm_joint_ids] = env._arm_upper
        actions_module.apply_action_pipeline(env, torch.full((2, 54), 3.))
        torch.testing.assert_close(env._cur_targets[:, env._arm_joint_ids], env._arm_upper.expand(2, -1))
        self.assertTrue((env._cur_targets[:, env._hand_joint_ids] <= env._hand_upper).all())
        self.assertTrue((env._cur_targets[:, env._hand_joint_ids] >= env._hand_lower).all())
        env._prev_targets[:, env._arm_joint_ids] = env._arm_lower
        actions_module.apply_action_pipeline(env, torch.full((2, 54), -3.))
        torch.testing.assert_close(env._cur_targets[:, env._arm_joint_ids], env._arm_lower.expand(2, -1))

    def test_partial_episode_start_flushes_only_its_delay_queue(self):
        env, reference = self.make_env(delay=True), self.make_env()
        env._action_queue.fill_(123)
        env.episode_length_buf[1] = 10
        values = torch.linspace(-1, 1, 54).repeat(2, 1)
        actions_module.apply_action_pipeline(env, values)
        actions_module.apply_action_pipeline(reference, values)
        torch.testing.assert_close(env._cur_targets[0], reference._cur_targets[0])
        mapped = values[:, env._perm_canon_to_lab]
        torch.testing.assert_close(env._action_queue[0], mapped[0].expand(3, -1))
        torch.testing.assert_close(env._action_queue[1, 0], mapped[1])
        self.assertTrue((env._action_queue[1, 1:] == 123).all())

    def test_rejects_single_hand_action_shape(self):
        with self.assertRaisesRegex(ValueError, "54"):
            actions_module.apply_action_pipeline(self.make_env(), torch.zeros(2, 27))


if __name__ == "__main__":
    unittest.main()
