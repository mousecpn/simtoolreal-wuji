"""rl_games adapter for the KUKA/Sharpa world-frame SE3 action contract."""

from __future__ import annotations

import torch
from torch import nn

from .c4 import (
    C4ActorCritic,
    EquivariantLinear,
    mixed_type,
    quaternion_columns,
    regular_type,
    export_for_rollout,
)

# (raw width, representation). Quaternions in legacy policy observations: xyzw.
FIELDS = {
    "joint_pos": (29, "scalar"),
    "joint_vel": (29, "scalar"),
    "prev_action_targets": (29, "scalar"),
    "object_scales": (3, "scalar"),
    "palm_pos": (3, "vector"),
    "object_pos": (3, "vector"),
    "goal_pos": (3, "vector"),
    "palm_rot": (4, "quat"),
    "object_rot": (4, "quat"),
    "goal_rot": (4, "quat"),
    "palm_vel": (6, "vector"),
    "object_vel": (6, "vector"),
    "fingertip_pos_rel_palm": (15, "vector"),
    "keypoints_rel_palm": (12, "vector"),
    "keypoints_rel_goal": (12, "vector"),
    "bbox_coords": (24, "vector"),
    "closest_keypoint_max_dist": (1, "scalar"),
    "closest_fingertip_dist": (5, "scalar"),
    "lifted_object": (1, "scalar"),
    "progress": (1, "scalar"),
    "successes": (1, "scalar"),
    "reward": (1, "scalar"),
}


class ObservationEncoder(nn.Module):
    def __init__(self, fields):
        super().__init__()
        self.fields = tuple(fields)
        self.widths = [FIELDS[f][0] for f in fields]
        self.raw_size = sum(self.widths)
        self.invariant_channels = sum(
            FIELDS[f][0] for f in fields if FIELDS[f][1] == "scalar"
        )
        self.num_vectors = sum(
            2 if FIELDS[f][1] == "quat" else FIELDS[f][0] // 3
            for f in fields
            if FIELDS[f][1] != "scalar"
        )
        self.out_type = mixed_type(self.invariant_channels, self.num_vectors)

    def forward(self, obs):
        if obs.shape[-1] != self.raw_size:
            raise ValueError(
                f"expected {self.raw_size} observations, got {obs.shape[-1]}"
            )
        scalars, vectors = [], []
        for name, part in zip(self.fields, obs.split(self.widths, -1)):
            kind = FIELDS[name][1]
            if kind == "scalar":
                scalars.append(part)
            elif kind == "quat":
                vectors.append(quaternion_columns(part[..., [3, 0, 1, 2]]))
            else:
                vectors.append(part.reshape(*part.shape[:-1], -1, 3))
        # XYZ remains three numbers: XY is irrep(1), Z is irrep(0).
        # FieldType has the same order as this concatenation; no orbit expansion.
        return torch.cat(
            (torch.cat(scalars, -1), torch.cat(vectors, -2).flatten(-2)), -1
        )


class ExplorationConditioning(nn.Module):
    """SAPG learned block embeddings are invariant scalars under C4 rotations."""

    def __init__(self, encoder, **kwargs):
        super().__init__()
        if kwargs.get("coef_id_idx") != encoder.raw_size:
            raise ValueError("SAPG coefficient ID must follow the raw observations")
        if tuple(kwargs["input_shape"]) != (encoder.raw_size + 1,):
            raise ValueError("SAPG expects raw observations plus one coefficient ID")
        ids = kwargs["coef_ids"].detach().clone()
        if ids.ndim != 1 or not len(ids) or len(ids.unique()) != len(ids):
            raise ValueError("SAPG requires distinct coefficient IDs")
        self.register_buffer("coef_ids", ids)
        self.raw_size = encoder.raw_size
        self.scalar_size = encoder.invariant_channels
        self.channels = kwargs.get("param_size", 32)
        self.extra_params = nn.Parameter(torch.randn(len(ids), self.channels))

    def forward(self, obs, encoder):
        indices = (
            (obs[..., self.raw_size, None] == self.coef_ids)
            .to(torch.int64).argmax(-1)
        )
        features = encoder(obs[..., :self.raw_size])
        # The encoder orders invariant scalars before vectors. Keep that order.
        features = torch.cat(
            (features[..., :self.scalar_size], self.extra_params[indices],
             features[..., self.scalar_size:]), -1,
        )
        return features, indices


class TorchRolloutMixin:
    """Use a frozen Torch snapshot for no-grad sampling, refreshing after updates."""

    def _init_torch_rollout(self, params):
        self.torch_rollout = params.get("torch_rollout", False)
        self.__dict__["_torch_rollout_snapshot"] = None
        self.__dict__["_torch_rollout_versions"] = None
        self.register_load_state_dict_post_hook(self._clear_torch_rollout)

    @staticmethod
    def _clear_torch_rollout(module, incompatible_keys):
        module.__dict__["_torch_rollout_snapshot"] = None
        module.__dict__["_torch_rollout_versions"] = None

    def train(self, mode=True):
        if mode:
            self._clear_torch_rollout(self, None)
        elif getattr(self, "torch_rollout", False):
            versions = tuple(
                (id(p), p._version, p.device, p.dtype) for p in self.parameters()
            )
            if self.__dict__.get("_torch_rollout_versions") != versions:
                self.__dict__["_torch_rollout_snapshot"] = None
                self.__dict__["_torch_rollout_versions"] = versions
        return super().train(mode)

    def export_for_rollout(self):
        return export_for_rollout(self)

    def _apply(self, fn, recurse=True):
        self._clear_torch_rollout(self, None)
        return super()._apply(fn, recurse=recurse)

    def _get_torch_rollout(self):
        if self.training or not self.torch_rollout or torch.is_grad_enabled():
            return None
        snapshot = self.__dict__.get("_torch_rollout_snapshot")
        if snapshot is None:
            snapshot = self.export_for_rollout()
            # Keep snapshots outside registered modules/optimizer/checkpoints.
            self.__dict__["_torch_rollout_snapshot"] = snapshot
        return snapshot


class C4Network(TorchRolloutMixin, nn.Module):
    def __init__(self, params, **kwargs):
        super().__init__()
        if kwargs.get("normalize_input", False):
            raise ValueError(
                "C4 requires normalize_input=false: coordinate-wise running statistics break equivariance"
            )
        if kwargs["actions_num"] != 28 or kwargs.get("value_size", 1) != 1:
            raise ValueError("C4 KUKA/Sharpa requires 28 actions and one value")
        self.encoder = ObservationEncoder(params["obs_fields"])
        self.conditioning = (
            ExplorationConditioning(self.encoder, **kwargs)
            if kwargs.get("type") == "extra_param" else None
        )
        if self.conditioning is None and tuple(kwargs["input_shape"]) != (
            self.encoder.raw_size,
        ):
            raise ValueError(
                "network obs_fields must match the environment obs_list order and width"
            )
        self.num_seqs = kwargs.get("num_seqs", 1)
        self.policy = C4ActorCritic(
            self.encoder.invariant_channels
            + (self.conditioning.channels if self.conditioning else 0),
            self.encoder.num_vectors,
            22,
            params.get("hidden_channels", 64),
            params.get("layers", 1),
            input_channels=params.get("input_channels"),
            mlp_units=params.get("mlp_units", ()),
            layer_norm=params.get("layer_norm", False),
            log_std_groups=len(self.conditioning.coef_ids) if self.conditioning else 0,
        )
        self._init_torch_rollout(params)

    def is_rnn(self):
        return True

    def is_separate_critic(self):
        return False

    def get_value_layer(self):
        return self.policy.critic

    def get_default_rnn_state(self):
        p = next(self.parameters())
        rnn = self.policy.rnn
        # rl_games expects (layers, batch, features), packing the group axis.
        shape = (rnn.num_layers, self.num_seqs, 4 * rnn.hidden_channels)
        return p.new_zeros(shape), p.new_zeros(shape)

    def forward(self, obs_dict):
        snapshot = self._get_torch_rollout()
        if snapshot is not None:
            return snapshot(obs_dict)
        obs = obs_dict["obs"]
        length = obs_dict.get("seq_length", 1)
        batch = obs.shape[0] // length
        obs = obs.reshape(batch, length, -1).transpose(0, 1)
        if self.conditioning is None:
            features, coef_indices = self.encoder(obs), None
        else:
            features, coef_indices = self.conditioning(obs, self.encoder)
        states = obs_dict.get("rnn_states")
        rnn = self.policy.rnn
        if states is not None:
            states = tuple(
                s.reshape(rnn.num_layers, batch, 4, rnn.hidden_channels) for s in states
            )
        dones = obs_dict.get("dones")
        if dones is not None:
            dones = dones.reshape(batch, length).transpose(0, 1)
        mean, std, value, states = self.policy(features, states, dones, coef_indices)

        def flatten(y):
            return y.transpose(0, 1).reshape(batch * length, -1)

        return (
            flatten(mean),
            flatten(std),
            flatten(value),
            tuple(s.flatten(-2) for s in states),
        )


class C4CentralValueNetwork(TorchRolloutMixin, nn.Module):
    """Independent privileged-state MLP; scalar output is C4 invariant.

    Same scalar MLP widths/ELU/order as the baseline central value network.
    The first escnn layer maps mixed irreps directly into regular features.
    No recurrent state or parameters are shared with the actor network.
    """

    def __init__(self, params, **kwargs):
        super().__init__()
        if kwargs.get("normalize_input", False):
            raise ValueError("C4 central critic requires normalize_input=false")
        if kwargs.get("value_size", 1) != 1:
            raise ValueError("C4 central critic requires one value")
        self.encoder = ObservationEncoder(params["obs_fields"])
        self.conditioning = (
            ExplorationConditioning(self.encoder, **kwargs)
            if kwargs.get("type") == "extra_param" else None
        )
        if self.conditioning is None and tuple(kwargs["input_shape"]) != (
            self.encoder.raw_size,
        ):
            raise ValueError(
                "central critic obs_fields must match environment state_list"
            )
        field_type = (
            mixed_type(
                self.encoder.invariant_channels + self.conditioning.channels,
                self.encoder.num_vectors,
            )
            if self.conditioning else self.encoder.out_type
        )
        modules = []
        for width in params["mlp_units"]:
            if width < 4 or width % 4:
                raise ValueError(
                    "regular MLP scalar widths must be positive multiples of 4"
                )
            out_type = regular_type(width // 4)
            modules.extend((EquivariantLinear(field_type, out_type), nn.ELU()))
            field_type = out_type
        self.mlp = nn.Sequential(*modules)
        self.value = EquivariantLinear(field_type, mixed_type(1, 0))
        self._init_torch_rollout(params)

    def is_rnn(self):
        return False

    def is_separate_critic(self):
        return False

    def get_default_rnn_state(self):
        return None

    def get_value_layer(self):
        return self.value

    def forward(self, obs_dict):
        snapshot = self._get_torch_rollout()
        if snapshot is not None:
            return snapshot(obs_dict)
        obs = obs_dict["obs"]
        features = (
            self.encoder(obs) if self.conditioning is None
            else self.conditioning(obs, self.encoder)[0]
        )
        return self.value(self.mlp(features)), None


class C4Builder:
    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        if self.params.get("central_value", False):
            return C4CentralValueNetwork(self.params, **kwargs)
        return C4Network(self.params, **kwargs)
