"""escnn C4 linear maps and an LSTM with regular hidden/cell states.

Regular tensors use (..., 4, channels) externally. escnn packs each regular
field contiguously as (..., channels, 4); conversion is explicit at boundaries.
The generator is an active +90 degree rotation about world Z, acting on the
regular group axis as torch.roll(x, 1, dims=-2).
"""

from __future__ import annotations

import torch
from escnn import group, gspaces
from escnn import nn as enn
from torch import nn
from torch.nn import functional as F

C4 = group.cyclic_group(4)
C4_SPACE = gspaces.no_base_space(C4)


def regular_type(channels: int) -> enn.FieldType:
    return enn.FieldType(C4_SPACE, [C4.regular_representation] * channels)


def mixed_type(invariant_channels: int, num_vectors: int) -> enn.FieldType:
    """[invariant scalars, (x,y,z), ...], with each XYZ = irrep(1) + irrep(0)."""
    return enn.FieldType(
        C4_SPACE,
        [C4.irrep(0)] * invariant_channels
        + [rep for _ in range(num_vectors) for rep in (C4.irrep(1), C4.irrep(0))],
    )


def pack_regular(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(-1, -2).flatten(-2)


def unpack_regular(x: torch.Tensor, channels: int) -> torch.Tensor:
    return x.reshape(*x.shape[:-1], channels, 4).transpose(-1, -2)


class EquivariantLinear(enn.Linear):
    """escnn Linear accepting flat tensors with arbitrary batch dimensions.

    Expanded eval matrices are derived caches, excluded from checkpoints. This
    gives the same checkpoint format in train/eval, and refreshes eval caches
    after loading learned coefficients (including loads into an existing model).
    """

    def __init__(self, in_type, out_type, bias=True):
        super().__init__(in_type, out_type, bias=bias)
        self._non_persistent_buffers_set.update(("matrix", "expanded_bias"))
        self.register_load_state_dict_post_hook(self._refresh_eval_cache)

    def train(self, mode=True):
        result = super().train(mode)
        self._non_persistent_buffers_set.update(("matrix", "expanded_bias"))
        return result

    @staticmethod
    def _refresh_eval_cache(module, incompatible_keys):
        if not module.training:
            with torch.no_grad():
                module.train(False)

    def forward(self, x):
        if isinstance(x, enn.GeometricTensor):
            return super().forward(x)
        shape = x.shape[:-1]
        y = super().forward(
            enn.GeometricTensor(x.reshape(-1, self.in_type.size), self.in_type)
        )
        return y.tensor.reshape(*shape, self.out_type.size)


def rotations(like: torch.Tensor) -> torch.Tensor:
    return like.new_tensor(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
            [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
        ]
    )


def quaternion_columns(q: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion -> first two rotation columns, shape (..., 2, 3).

    Using rotation columns removes the quaternion double-cover/sign ambiguity.
    """
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y + w * z),
            2 * (x * z - w * y),
            2 * (x * y - w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z + w * x),
        ),
        -1,
    ).reshape(*q.shape[:-1], 2, 3)


class C4Linear(nn.Module):
    """escnn regular-to-regular Linear with an explicit group axis."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.linear = EquivariantLinear(
            regular_type(in_channels), regular_type(out_channels)
        )

    def forward(self, x):
        return unpack_regular(self.linear(pack_regular(x)), self.out_channels)


class C4LSTM(nn.Module):
    """Multi-layer escnn LSTM; both h and c carry the regular representation.

    Input/output: (time, batch, 4, channels). State: two tensors of shape
    (layers, batch, 4, hidden_channels). dones[t,b] resets BEFORE input t.
    Gate preactivations use escnn Linear; regular fields permit pointwise
    sigmoid/tanh and products since C4 acts by permutations on their slots.
    """

    def __init__(self, input_channels: int, hidden_channels: int, num_layers: int = 1):
        super().__init__()
        if min(input_channels, hidden_channels, num_layers) < 1:
            raise ValueError("channel and layer counts must be positive")
        self.hidden_channels, self.num_layers = hidden_channels, num_layers
        self.state_type = regular_type(hidden_channels)
        self.inputs = nn.ModuleList(
            C4Linear(input_channels if i == 0 else hidden_channels, 4 * hidden_channels)
            for i in range(num_layers)
        )
        self.recurrents = nn.ModuleList(
            C4Linear(hidden_channels, 4 * hidden_channels) for _ in range(num_layers)
        )

    def forward(self, x, state=None, dones=None):
        if x.ndim != 4 or x.shape[-2] != 4 or x.shape[0] == 0:
            raise ValueError("expected nonempty (time, batch, 4, channels) input")
        shape = (self.num_layers, x.shape[1], 4, self.hidden_channels)
        if state is None:
            state = (x.new_zeros(shape), x.new_zeros(shape))
        if len(state) != 2 or any(tuple(s.shape) != shape for s in state):
            raise ValueError(f"expected h,c shapes {shape}")
        h, c = list(state[0].unbind(0)), list(state[1].unbind(0))
        outputs = []
        for t in range(x.shape[0]):
            y = x[t]
            for i, (inp, rec) in enumerate(zip(self.inputs, self.recurrents)):
                if dones is not None:
                    keep = (~dones[t].bool()).to(x.dtype).reshape(-1, 1, 1)
                    h[i], c[i] = h[i] * keep, c[i] * keep
                a, b, u, o = (inp(y) + rec(h[i])).chunk(4, -1)
                c[i] = b.sigmoid() * c[i] + a.sigmoid() * u.tanh()
                h[i] = o.sigmoid() * c[i].tanh()
                y = h[i]
            outputs.append(y)
        return torch.stack(outputs), (torch.stack(h), torch.stack(c))


class C4LayerNorm(nn.Module):
    """LayerNorm over all regular slots, with affine parameters tied per field."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels, self.eps = channels, eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        y = F.layer_norm(pack_regular(x), (4 * self.channels,), eps=self.eps)
        return unpack_regular(y, self.channels) * self.weight + self.bias


class C4ActorCritic(nn.Module):
    """Mixed irreps -> escnn Linear -> regular LSTM -> typed action/value heads.

    World-frame twist vectors transform as irrep(1)+irrep(0); joint actions
    and value are trivial. XY Gaussian standard deviations are tied per vector.
    """

    def __init__(
        self,
        invariant_channels,
        num_vectors,
        joint_actions,
        hidden_channels=64,
        num_layers=1,
        input_channels=None,
        mlp_units=(),
        layer_norm=False,
    ):
        super().__init__()
        self.input_type = mixed_type(invariant_channels, num_vectors)
        self.input_channels = input_channels or hidden_channels
        self.latent_type = regular_type(self.input_channels)
        self.output_type = mixed_type(0, 2) + enn.FieldType(
            C4_SPACE, [C4.irrep(0)] * joint_actions
        )
        self.input_projection = EquivariantLinear(self.input_type, self.latent_type)
        self.rnn = C4LSTM(self.input_channels, hidden_channels, num_layers)
        self.layer_norm = C4LayerNorm(hidden_channels) if layer_norm else nn.Identity()
        modules = []
        channels = hidden_channels
        for width in mlp_units:
            if width < 4 or width % 4:
                raise ValueError(
                    "regular MLP scalar widths must be positive multiples of 4"
                )
            modules.extend((C4Linear(channels, width // 4), nn.ELU()))
            channels = width // 4
        self.mlp = nn.Sequential(*modules)
        head_type = regular_type(channels)
        self.actor = EquivariantLinear(head_type, self.output_type)
        self.critic = EquivariantLinear(
            head_type, enn.FieldType(C4_SPACE, [C4.irrep(0)])
        )
        self.log_std = nn.Parameter(torch.zeros(4 + joint_actions))

    def forward(self, features, state=None, dones=None):
        x = unpack_regular(self.input_projection(features), self.input_channels)
        y, state = self.rnn(x, state, dones)
        y = pack_regular(self.mlp(self.layer_norm(y)))
        mean = self.actor(y)
        idx = [0, 0, 1, 2, 2, 3] + list(range(4, self.log_std.numel()))
        log_std = self.log_std[idx].expand_as(mean)
        value = self.critic(y)
        return mean, log_std, value, state
