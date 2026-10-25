"""escnn C4 linear maps and an LSTM with regular hidden/cell states.

Regular tensors use (..., 4, channels) externally. escnn packs each regular
field contiguously as (..., channels, 4); conversion is explicit at boundaries.
The generator is an active +90 degree rotation about world Z, acting on the
regular group axis as torch.roll(x, 1, dims=-2).
"""

from __future__ import annotations

import copy

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
        self._eval_weight_versions = None
        self._non_persistent_buffers_set.update(("matrix", "expanded_bias"))
        self.register_load_state_dict_post_hook(self._refresh_eval_cache)

    def train(self, mode=True):
        # rl_games calls eval() on every actor/critic query. escnn otherwise
        # expands the basis again on every such call, even while already eval.
        versions = (
            id(self.weights), self.weights._version,
            self.weights.device, self.weights.dtype,
            (id(self.bias), self.bias._version) if self.bias is not None else None,
        )
        if (
            not mode and not self.training
            and self._eval_weight_versions == versions
            and hasattr(self, "matrix")
        ):
            return self
        result = super().train(mode)
        self._eval_weight_versions = None if mode else versions
        self._non_persistent_buffers_set.update(("matrix", "expanded_bias"))
        return result

    @staticmethod
    def _refresh_eval_cache(module, incompatible_keys):
        if not module.training:
            with torch.no_grad():
                module._eval_weight_versions = None
                module.train(False)

    def forward(self, x):
        if isinstance(x, enn.GeometricTensor):
            return super().forward(x)
        shape = x.shape[:-1]
        if not self.training:
            # Cached equivariant weights are ordinary dense Torch matrices.
            return F.linear(
                x.reshape(-1, self.in_type.size), self.matrix, self.expanded_bias
            ).reshape(*shape, self.out_type.size)
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
        log_std_groups=0,
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
        std_shape = (
            (log_std_groups, 4 + joint_actions)
            if log_std_groups else (4 + joint_actions,)
        )
        self.log_std = nn.Parameter(torch.zeros(std_shape))

    def forward(self, features, state=None, dones=None, coef_indices=None):
        x = unpack_regular(self.input_projection(features), self.input_channels)
        y, state = self.rnn(x, state, dones)
        y = pack_regular(self.mlp(self.layer_norm(y)))
        mean = self.actor(y)
        idx = [0, 0, 1, 2, 2, 3] + list(range(4, self.log_std.shape[-1]))
        std = self.log_std if coef_indices is None else self.log_std[coef_indices]
        log_std = std[..., idx].expand_as(mean)
        value = self.critic(y)
        return mean, log_std, value, state


class TorchLSTM(nn.Module):
    """Frozen nn.LSTM snapshot with the C4 LSTM's external state layout."""

    def __init__(self, source: C4LSTM):
        super().__init__()
        self.hidden_channels = source.hidden_channels
        self.num_layers = source.num_layers
        self.rnn = nn.LSTM(
            source.inputs[0].linear.in_type.size,
            4 * self.hidden_channels,
            self.num_layers,
            device="meta",
        )
        # escnn's field-major output is already (i,f,g,o) x (channel,group),
        # exactly PyTorch's gate order when hidden features use field-major.
        # The C4 external state uses (group,channel), converted at the boundary.
        for layer, (inp, rec) in enumerate(zip(source.inputs, source.recurrents)):
            for suffix, linear in (("ih", inp.linear), ("hh", rec.linear)):
                weight, bias = linear.expand_parameters()
                setattr(self.rnn, f"weight_{suffix}_l{layer}",
                        nn.Parameter(weight.detach().clone(), requires_grad=False))
                setattr(self.rnn, f"bias_{suffix}_l{layer}",
                        nn.Parameter(bias.detach().clone(), requires_grad=False))
        self.rnn.flatten_parameters()

    def forward(self, x, state=None, dones=None):
        x = pack_regular(x)
        if state is not None:
            state = tuple(pack_regular(s) for s in state)
        if dones is None:
            y, state = self.rnn(x, state)
        else:
            if state is None:
                shape = (self.num_layers, x.shape[1], 4 * self.hidden_channels)
                state = (x.new_zeros(shape), x.new_zeros(shape))
            outputs = []
            for t in range(x.shape[0]):
                keep = (~dones[t].bool()).to(x.dtype).reshape(1, -1, 1)
                state = tuple(s * keep for s in state)
                y, state = self.rnn(x[t:t + 1], state)
                outputs.append(y)
            y = torch.cat(outputs, 0)
        return unpack_regular(y, self.hidden_channels), tuple(
            unpack_regular(s, self.hidden_channels) for s in state
        )


@torch.no_grad()
def export_for_rollout(module: nn.Module) -> nn.Module:
    """Snapshot a C4 network as frozen dense Torch layers and fused nn.LSTM.

    Training parameters, optimizer references, mode and RNG state are unchanged.
    Export again after each update if using a separate rollout snapshot.
    """
    memo = {}

    def replace(child):
        if id(child) in memo:
            return
        if isinstance(child, C4LSTM):
            memo[id(child)] = TorchLSTM(child)
        elif isinstance(child, EquivariantLinear):
            weight, bias = child.expand_parameters()
            linear = nn.Linear(
                child.in_type.size, child.out_type.size,
                bias=bias is not None, device="meta",
            )
            linear.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
            if bias is not None:
                linear.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
            memo[id(child)] = linear
        else:
            snapshot = child.__dict__.get("_torch_rollout_snapshot")
            if snapshot is not None:
                memo[id(snapshot)] = None
            for descendant in child.children():
                replace(descendant)

    replace(module)
    exported = copy.deepcopy(module, memo)
    for child in exported.modules():
        if hasattr(child, "torch_rollout"):
            child.torch_rollout = False
    return exported.requires_grad_(False).eval()
