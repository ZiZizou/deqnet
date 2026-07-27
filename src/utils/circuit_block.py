import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Union, Optional, Any, Tuple, Callable
import numpy as np
import torch.nn.init as init
from .baseline import trace_jac, FeedForwardNet
import warnings
from math import ceil
import sys


_MONOTONE_ACTIVATIONS = {'relu', 'leaky_relu', 'tanh', 'sigmoid', 'elu', 'softplus'}


class Device(nn.Module):
    def __init__(self, num_edge, negation, activation, max_gain=10.0, reparam=True):
        super(Device, self).__init__()
        if not reparam:
            warnings.warn(
                "Device with reparam=False is legacy ODE-only and does NOT satisfy "
                "translation invariance / passivity.  Do not use in DEQ mode.",
                DeprecationWarning,
            )
        self.num_edge = num_edge
        self.max_gain = float(max_gain)
        self.reparam = bool(reparam)
        if activation not in _MONOTONE_ACTIVATIONS:
            warnings.warn(
                f"activation={activation!r} is not in the DEQ monotone whitelist "
                f"{_MONOTONE_ACTIVATIONS}; passivity certificate will be conservative.",
                UserWarning,
            )
        self.activation_name = activation
        self.activation_function = (
            getattr(F, activation) if activation is not None else lambda x: x
        )
        if negation:
            self.raw_gain = nn.Parameter(torch.zeros(self.num_edge))
            self.bias = nn.Parameter(torch.zeros(self.num_edge))
            if not self.reparam:
                self.param = nn.Parameter(torch.stack([self.bias.detach(), torch.zeros(self.num_edge)], dim=0))
        else:
            self.raw_gain = nn.Parameter(torch.zeros(self.num_edge))
            self.raw_gain_b = nn.Parameter(torch.zeros(self.num_edge))
            self.bias = nn.Parameter(torch.zeros(self.num_edge))
            if not self.reparam:
                self.param = nn.Parameter(torch.zeros(3, self.num_edge))
        self.negation = negation

    @property
    def gain(self):
        if self.reparam:
            return self.max_gain * torch.sigmoid(self.raw_gain)
        if self.negation:
            return self.param[1]
        return self.param[1]

    def max_slope(self):
        g = self.gain.detach()
        if self.activation_name == 'sigmoid':
            return g / 4.0
        return g

    def forward(self, x_src, x_des):
        if self.negation:
            x = x_src - x_des
            res = self.activation_function(self.gain * x + self.bias)
        else:
            if self.reparam:
                res = self.activation_function(self.gain * x_src + self.gain * x_des + self.bias)
            else:
                res = self.activation_function(x_src * self.param[0] + x_des * self.param[1] + self.param[2])
        return res


class ShiftRelu1(nn.Module):
    def __init__(self, num_edge, max_gain=10.0, reparam=True):
        super(ShiftRelu1, self).__init__()
        self.num_edge = num_edge
        self.max_gain = float(max_gain)
        self.reparam = bool(reparam)
        self.raw_gain = nn.Parameter(torch.zeros(num_edge))
        self.bias = nn.Parameter(torch.zeros(num_edge))
        if not self.reparam:
            self.param = nn.Parameter(torch.stack([self.bias.detach(), torch.zeros(num_edge)], dim=0))

    @property
    def gain(self):
        if self.reparam:
            return self.max_gain * torch.sigmoid(self.raw_gain)
        return self.param[1]

    def max_slope(self):
        return self.gain.detach()

    def forward(self, x_src, x_des):
        x = x_src - x_des
        res = self.gain * F.relu(x - self.bias)
        return res


class ShiftLeakyRelu1(nn.Module):
    def __init__(self, num_edge, max_gain=10.0, reparam=True):
        super(ShiftLeakyRelu1, self).__init__()
        self.num_edge = num_edge
        self.max_gain = float(max_gain)
        self.reparam = bool(reparam)
        self.raw_gain = nn.Parameter(torch.zeros(num_edge))
        self.bias = nn.Parameter(torch.zeros(num_edge))
        if not self.reparam:
            self.param = nn.Parameter(torch.stack([self.bias.detach(), torch.zeros(num_edge)], dim=0))

    @property
    def gain(self):
        if self.reparam:
            return self.max_gain * torch.sigmoid(self.raw_gain)
        return self.param[1]

    def max_slope(self):
        return self.gain.detach()

    def forward(self, x_src, x_des):
        x = x_src - x_des
        res = self.gain * F.leaky_relu(x - self.bias)
        return res


class ShiftRelu2(nn.Module):
    def __init__(self, num_edge, max_gain=10.0, reparam=True):
        super(ShiftRelu2, self).__init__()
        self.num_edge = num_edge
        self.max_gain = float(max_gain)
        self.reparam = bool(reparam)
        self.raw_gain_src = nn.Parameter(torch.zeros(num_edge))
        self.raw_gain_des = nn.Parameter(torch.zeros(num_edge))
        self.bias = nn.Parameter(torch.zeros(num_edge))
        if not self.reparam:
            self.param = nn.Parameter(torch.zeros(3, num_edge))

    def forward(self, x_src, x_des):
        if self.reparam:
            gs = self.max_gain * torch.sigmoid(self.raw_gain_src)
            gd = self.max_gain * torch.sigmoid(self.raw_gain_des)
            res = F.relu(gs * x_src + gd * x_des + self.bias)
        else:
            res = F.relu(x_src * self.param[0] + x_des * self.param[1] + self.param[2])
        return res


class ShiftLeakyRelu2(nn.Module):
    def __init__(self, num_edge, max_gain=10.0, reparam=True):
        super(ShiftLeakyRelu2, self).__init__()
        self.num_edge = num_edge
        self.max_gain = float(max_gain)
        self.reparam = bool(reparam)
        self.raw_gain_src = nn.Parameter(torch.zeros(num_edge))
        self.raw_gain_des = nn.Parameter(torch.zeros(num_edge))
        self.bias = nn.Parameter(torch.zeros(num_edge))
        if not self.reparam:
            self.param = nn.Parameter(torch.zeros(3, num_edge))

    def forward(self, x_src, x_des):
        if self.reparam:
            gs = self.max_gain * torch.sigmoid(self.raw_gain_src)
            gd = self.max_gain * torch.sigmoid(self.raw_gain_des)
            res = F.leaky_relu(gs * x_src + gd * x_des + self.bias)
        else:
            res = F.leaky_relu(x_src * self.param[0] + x_des * self.param[1] + self.param[2])
        return res


class ShiftTanh1(nn.Module):
    def __init__(self, num_edge, max_gain=10.0, reparam=True):
        super(ShiftTanh1, self).__init__()
        self.num_edge = num_edge
        self.max_gain = float(max_gain)
        self.reparam = bool(reparam)
        self.raw_gain = nn.Parameter(torch.zeros(num_edge))
        self.bias = nn.Parameter(torch.zeros(num_edge))
        if not self.reparam:
            self.param = nn.Parameter(torch.stack([self.bias.detach(), torch.zeros(num_edge), torch.zeros(num_edge)], dim=0))

    @property
    def gain(self):
        if self.reparam:
            return self.max_gain * torch.sigmoid(self.raw_gain)
        return self.param[1]

    def max_slope(self):
        return self.gain.detach()

    def forward(self, x_src, x_des):
        x = x_src - x_des
        res = F.tanh(self.gain * x + self.bias)
        return res


class ShiftTanh2(nn.Module):
    def __init__(self, num_edge, max_gain=10.0, reparam=True):
        super(ShiftTanh2, self).__init__()
        self.num_edge = num_edge
        self.max_gain = float(max_gain)
        self.reparam = bool(reparam)
        self.raw_gain_src = nn.Parameter(torch.zeros(num_edge))
        self.raw_gain_des = nn.Parameter(torch.zeros(num_edge))
        self.bias = nn.Parameter(torch.zeros(num_edge))
        if not self.reparam:
            self.param = nn.Parameter(torch.zeros(3, num_edge))

    def forward(self, x_src, x_des):
        if self.reparam:
            gs = self.max_gain * torch.sigmoid(self.raw_gain_src)
            gd = self.max_gain * torch.sigmoid(self.raw_gain_des)
            res = F.tanh(gs * x_src + gd * x_des + self.bias)
        else:
            res = F.tanh(x_src * self.param[0] + x_des * self.param[1] + self.param[2])
        return res


class Conductance(nn.Module):
    def __init__(self, num_edge, max_gain=10.0, reparam=True):
        super(Conductance, self).__init__()
        self.num_edge = num_edge
        self.max_gain = float(max_gain)
        self.reparam = bool(reparam)
        self.raw_gain = nn.Parameter(torch.zeros(num_edge))
        if not self.reparam:
            self.param = nn.Parameter(torch.zeros(3, num_edge))

    @property
    def gain(self):
        if self.reparam:
            return self.max_gain * torch.sigmoid(self.raw_gain)
        return self.param[0]

    def max_slope(self):
        return self.gain.detach()

    def forward(self, x_src, x_des):
        x = x_src - x_des
        res = self.gain * x
        return res


def _preprocess_net_topo(net_topo: Union[List, Tuple, np.ndarray, torch.Tensor]) -> Tuple[
    torch.Tensor, torch.Tensor, int, int]:
    if isinstance(net_topo, (list, tuple, np.ndarray)):
        net_topo = torch.Tensor(net_topo).to(torch.int64)
    elif isinstance(net_topo, torch.Tensor):
        net_topo = net_topo.to(torch.int64)
    else:
        raise ValueError(f"Unsupported type of net_topo: {type(net_topo)}")

    if not (len(net_topo.shape) == 2 and net_topo.shape[1] >= 2):
        raise ValueError(f"Unsupported shape of net_topo, expected it to be (N, >=2), got {net_topo.shape}")

    return (net_topo[:, 0], net_topo[:, 1], int(torch.max(net_topo[:, :2]).item()), net_topo.shape[0])

def _preprocess_sim_dict(sim_dict: Dict[str, Any]) -> Dict:
    sim_dict['t_end'] = [float(val) for val in sim_dict['t_end']]
    sim_dict['tol'] = float(sim_dict['tol'])
    sim_dict['min_step'] = float(sim_dict['min_step'])
    sim_dict['first_step'] = float(sim_dict['first_step'])
    sim_dict['step_size'] = float(sim_dict['step_size'])
    return sim_dict

def _divide_time_bins(anchors, time_grids):
    anchors = anchors
    if time_grids[-1] > anchors[-1]:
        raise ValueError(f"The last time grid {time_grids[-1]} is larger than the last anchor {anchors[-1]}")
    if time_grids[0] < anchors[0]:
       raise ValueError(f"The first time grid {time_grids[0]} is smaller than the first anchor {anchors[0]}")

    result = [[] for _ in range(len(anchors)-1)]
    for i in range(len(anchors) - 1):
        left, right = anchors[i], anchors[i+1]
        result[i].append(left)
        for j in range(len(time_grids)):
            if time_grids[j] > left and time_grids[j] < right:
                result[i].append(time_grids[j])
        result[i].append(right)
    return result

def _init_model_param(keyword, model):
    if keyword == 'uniform':
        init_function = init.uniform_
    elif keyword == 'zeros':
        init_function = init.zeros_
    elif keyword == 'ones':
        init_function = init.ones_
    elif keyword == 'xavier':
        init_function = init.xavier_normal_
    elif keyword == 'gauss':
        init_function = lambda x: init.normal_(x, mean = 0.0, std=0.01)
    elif keyword == 'kaiming':
        init_function = init.kaiming_normal_
    else:
        raise NotImplementedError(f"Unsupported initialization: {keyword}")

    if isinstance(model, nn.ParameterList):
        for param in model:
            init_function(param)
    elif isinstance(model, nn.Parameter):
        init_function(model)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


class CircuitLayer(nn.Module):
    def __init__(self, net_topo, net_dict):
        super(CircuitLayer, self).__init__()
        self.src_node, self.des_node, self.max_node_index, self.num_edge = _preprocess_net_topo(net_topo)
        self.model = getattr(sys.modules[__name__], net_dict['model']['name'])(self.num_edge, **net_dict['model']['args'])
        self.nfe = torch.tensor(0.0)
        self.raw_gamma = nn.Parameter(torch.tensor(0.0))
        self.register_buffer('input_map', None)
        self._gamma_floor = 1e-2

        if hasattr(self.model, 'param'):
            _init_model_param(net_dict['initialization'], self.model.param)

    @property
    def gamma(self):
        return F.softplus(self.raw_gamma) + self._gamma_floor

    def set_input_nodes(self, input_nodes, device_index=None):
        if input_nodes is None or len(input_nodes) == 0:
            self.input_map = None
            return
        n = self.max_node_index
        d_in = len(input_nodes)
        S = torch.zeros(n, d_in)
        for i, node in enumerate(input_nodes):
            if node < 1 or node > n:
                raise ValueError(f"input_node {node} out of range [1, {n}] (1-indexed from ground)")
            S[node - 1, i] = 1.0
        if device_index is not None and device_index >= 0 and torch.cuda.is_available():
            S = S.to(f'cuda:{device_index}')
        self.input_map = S

    def worst_case_D(self):
        return self.model.max_slope()

    def prepare(self, device: List[int]) -> None:
        # device = [-1] (CPU) or a list with elements all >=0 , e.g., [0,1,2,]
        if len(device) == 1 and device[0] == -1:
            self.src_indices_list = [self.src_node]
            self.des_indices_list = [self.des_node]
        elif len(device) >= 1 and min(device) >= 0:
            self.src_indices_list = [self.src_node.to(device_index) if device_index in device else None for device_index
                                     in range(max(device) + 1)]
            self.des_indices_list = [self.des_node.to(device_index) if device_index in device else None for device_index
                                     in range(max(device) + 1)]
        else:
            raise ValueError(f"Unsupported device: {device}")

    def forward(self, t, x):

        self.nfe += torch.tensor(1.0)

        # Calculate the RHS of the ODE. Circuit ODE: \dot{v} = f(v). The time variable t won't be used; just placeholder.
        src_node, des_node = self.src_indices_list[x.get_device()], self.des_indices_list[x.get_device()]

        aux_v = torch.cat((torch.zeros_like(x[..., :1]), x), dim=-1)
        state_i = self.model(aux_v[..., src_node], aux_v[..., des_node])

        # add dummy node for ground
        result = torch.cat((torch.zeros_like(x[..., :1]), torch.zeros_like(x)), dim=-1)

        # Subtract state_i from the source nodes and add it to the destination nodes
        result.scatter_add_(-1, src_node.expand_as(state_i), -state_i)
        result.scatter_add_(-1, des_node.expand_as(state_i), state_i)

        return result[..., 1:]

    def rhs(self, v, u=None):
        """Equilibrium residual f(v,u) = -B^T g(B v_hat) - Gamma v + S u.

        Returns the residual (zero only when v = v*).  Same shape as v.
        Avoids in-place scatter_add_ for clean autograd.  Result slots are
        determined by v.shape[-1] (last-dim), not max_node_index, so callers
        can pass v of any consistent shape.
        """
        src_node, des_node = self.src_indices_list[v.get_device()], self.des_indices_list[v.get_device()]
        aux_v = torch.cat((torch.zeros_like(v[..., :1]), v), dim=-1)
        state_i = self.model(aux_v[..., src_node], aux_v[..., des_node])
        # Build result sized to v (not max_node_index) so f = result[..., 1:]
        # matches v's shape.  This mirrors the legacy zeros_like(v) behavior.
        batch_shape = v.shape[:-1]
        dev = v.device
        dtype = v.dtype
        result = torch.zeros(*batch_shape, v.shape[-1] + 1, dtype=dtype, device=dev)
        result = torch.scatter_add(result, -1, src_node.expand_as(state_i), -state_i)
        result = torch.scatter_add(result, -1, des_node.expand_as(state_i), state_i)
        f = result[..., 1:]
        f = f - self.gamma * v
        if u is not None and self.input_map is not None:
            S = self.input_map
            if S.device != f.device:
                S = S.to(f.device)
            f = f + u @ S.t()
        return f


class AugCircuitLayer(CircuitLayer):
    def __init__(self, net_struct, net_dict):
        super(AugCircuitLayer, self).__init__(net_struct, net_dict)

    def forward(self, t, states):
        self.nfe += torch.tensor(1.0)
        x, logp_x = states[0], states[1]
        src_node, des_node = self.src_indices_list[x.get_device()], self.des_indices_list[x.get_device()]

        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            # Calculate the RHS of the ODE. Circuit ODE: \dot{v} = f(v). The time variable t won't be used; just placeholder.
            aux_v = torch.cat((torch.zeros_like(x[..., :1]), x), dim=-1)
            state_i = self.model(aux_v[..., src_node], aux_v[..., des_node])

            # add dummy node for ground
            result = torch.cat((torch.zeros_like(x[..., :1]), torch.zeros_like(x)), dim=-1)

            # Subtract state_i from the source nodes and add it to the destination nodes
            result.scatter_add_(-1, src_node.expand_as(state_i), -state_i)
            result.scatter_add_(-1, des_node.expand_as(state_i), state_i)

            dx_dt = result[..., 1:]
            dlogp_x_dt = -trace_jac(dx_dt, x).view(x.shape[0], 1)

            return (dx_dt, dlogp_x_dt)


class CircuitBlock(nn.Module):
    def __init__(self, layer_list, sim_dict, residual, odeint, fill):
        super(CircuitBlock, self).__init__()
        self.layer_list = nn.ModuleList(layer_list)
        self.sim_dict = _preprocess_sim_dict(sim_dict)
        self.residual = residual
        self.odeint = odeint
        self.fill = fill
        self.set_integration_time(self.sim_dict['t_end'])
        print(self.integration_time)
    def set_integration_time(self, t: List) -> None:
        self.integration_time = _divide_time_bins(self.sim_dict['t_end'], t)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], reverse=False, return_middle=False) -> Tuple[torch.Tensor, List]:
        index = range(0, len(self.layer_list)) if not reverse else range(len(self.layer_list) - 1, -1, -1)
        middle = []
        for i in index:
            integration_time = torch.Tensor(self.integration_time[i]).to(x[0].device if isinstance(x, tuple) else x.device)
            integration_time = integration_time - integration_time[0]
            integration_time = integration_time.flip(0) if reverse else integration_time

            # Remedies for dimension inconsistency, applied without any warnings.
            if x.shape[-1] < self.layer_list[i].max_node_index:
                warnings.warn(f"index = {i} dimension mismatch, received dim = {x.shape[-1]}, current layer max_node_index = {self.layer_list[i].max_node_index}")
                if self.fill == 'zeros' or self.fill == 'zero':
                    x = torch.cat((x, torch.zeros(x.shape[0], self.layer_list[i].max_node_index - x.shape[1]).to(x.device)), dim=-1)
                elif self.fill == 'repeat':
                    x = x.repeat(1, ceil(self.layer_list[i].max_node_index / x.shape[1]))[:, :self.layer_list[i].max_node_index]
                else:
                    raise NotImplementedError(f"Unsupported fill method: {self.fill}")
            elif x.shape[-1] > self.layer_list[i].max_node_index:
                warnings.warn(f"index = {i} dimension mismatch, received dim = {x.shape[-1]}, current layer max_node_index = {self.layer_list[i].max_node_index}")
                x = x[:, :self.layer_list[i].max_node_index]

            out = self.odeint(self.layer_list[i], x, integration_time, rtol=self.sim_dict['tol'],
                              method=self.sim_dict['method'],
                              atol=self.sim_dict['tol'],
                              options={'first_step': self.sim_dict['first_step'],
                                       'min_step': self.sim_dict['min_step'],
                                       'step_size': self.sim_dict['step_size']
                                       })
            x = out[-1]

            if return_middle:
                middle.append(out)

            if self.residual is not None:
                if i == self.residual[0]:
                    residual_store = out[-1]
                if i == self.residual[1]:
                    x = x + residual_store

        return (x, middle)

class AugCircuitBlock(nn.Module):
    def __init__(self, layer_list: List[CircuitLayer], sim_dict: Dict[str, Any], residual, odeint: Callable, fill: Optional[str]= 'zero'):
        super(AugCircuitBlock, self).__init__()
        self.layer_list = nn.ModuleList(layer_list)
        self.sim_dict = _preprocess_sim_dict(sim_dict)
        self.residual = residual
        self.odeint = odeint
        self.fill = fill
        self.set_integration_time(self.sim_dict['t_end'])

    def set_integration_time(self, t: List) -> None:
        self.integration_time = _divide_time_bins(self.sim_dict['t_end'], t)
        print(self.integration_time)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], reverse, return_middle) -> Tuple[torch.Tensor, List]:
        index = range(0, len(self.layer_list)) if not reverse else range(len(self.layer_list) - 1, -1, -1)
        middle = []
        for i in index:
            integration_time = torch.Tensor(self.integration_time[i]).to(x[0].device if isinstance(x, tuple) else x.device)
            integration_time = integration_time - integration_time[0]
            integration_time = integration_time.flip(0) if reverse else integration_time

            # Careful: Dimension inconsistency
            if x[0].shape[1] != self.layer_list[i].max_node_index:
                warnings.warn(f"Dimension mismatch, previous layer dim = {x[0].shape[1]}, current layer max_node_index = {self.layer_list[i].max_node_index}")
                if i == 0 and x[0].shape[1] < self.layer_list[i].max_node_index:
                    warnings.warn(f"This is at the input level, we will do repeat padding.")
                    x = (x[0].repeat(1, ceil(self.layer_list[i].max_node_index / x[0].shape[1]))[:, :self.layer_list[i].max_node_index], x[1])

            out = self.odeint(self.layer_list[i], x, integration_time, rtol=self.sim_dict['tol'],
                              method=self.sim_dict['method'],
                              atol=self.sim_dict['tol'],
                              options={'first_step': self.sim_dict['first_step'],
                                       'min_step': self.sim_dict['min_step'],
                                       'step_size': self.sim_dict['step_size']
                                       })
            x = tuple(ele[-1] for ele in out)
            if return_middle:
                middle.append(out)

        return (x, middle)


# =============================================================================
# Deep Equilibrium (DEQ) extensions
# =============================================================================
from .deq_solver import (
    anderson as _anderson,
    fixed_point as _fixed_point,
    solve_jacobian_transpose as _solve_jt,
    check_contraction as _check_contraction,
    ConvergenceWarning as _ConvergenceWarning,
)


class EquilibriumSolve(torch.autograd.Function):
    """Implicit-differentiation wrapper around an equilibrium solve.

    forward: solve f(v, u) = 0 via Anderson (or fixed-point fallback),
             starting from v0.  No autograd graph is built.

    backward: re-run rhs_fn with enable_grad, then call .backward(-y) on the
             residual f_, where y solves J^T y = grad_out.  This populates
             .grad on all parameters that participated in the closure.

    backward_mode:
        'exact'   — full CG/fixed-point on J^T y = grad_out  (default, expensive)
        'phantom' — one VJP step (biased but cheap, robust to ill-conditioned J)

    Class attribute `.last_info` holds the most recent forward's info dict,
    useful for logging from outside the autograd graph.
    """

    last_info = None

    @staticmethod
    def forward(ctx, rhs_fn, v0, u, solver_cfg):
        cfg = dict(solver_cfg) if solver_cfg is not None else {}
        method = cfg.pop('method', 'anderson')
        max_iter = cfg.pop('max_iter', 100)
        tol = cfg.pop('tol', 1e-6)
        fwd_kwargs = {k: cfg[k] for k in ('m', 'lam', 'beta') if k in cfg}
        with torch.no_grad():
            if method == 'anderson':
                v_star, info = _anderson(lambda v: rhs_fn(v, u), v0,
                                          max_iter=max_iter, tol=tol, **fwd_kwargs)
            elif method == 'fixedpoint':
                v_star, info = _fixed_point(lambda v: rhs_fn(v, u), v0,
                                             max_iter=max_iter, tol=tol, **fwd_kwargs)
            else:
                raise ValueError(f"Unknown solver method: {method!r}")
        # Detach + require_grad so the autograd graph can flow even when no
        # input tensor had requires_grad=True (typical for downstream users).
        ctx.rhs_fn = rhs_fn
        ctx.v_star = v_star
        ctx.u = u
        ctx.solver_cfg = solver_cfg if solver_cfg is not None else {}
        ctx.info = info
        EquilibriumSolve.last_info = info
        if not v_star.requires_grad:
            v_star = v_star.detach().requires_grad_(True)
        return v_star

    @staticmethod
    def backward(ctx, grad_out):
        rhs_fn = ctx.rhs_fn
        v = ctx.v_star
        u = ctx.u
        solver_cfg = ctx.solver_cfg
        mode = solver_cfg.get('backward_mode', 'exact')
        bt_tol = solver_cfg.get('backward_tol', solver_cfg.get('tol', 1e-6))
        bt_max = solver_cfg.get('backward_max_iter', solver_cfg.get('max_iter', 50))
        phantom_damp = float(solver_cfg.get('phantom_damp', 0.5))
        phantom_steps = int(solver_cfg.get('phantom_steps', 1))

        v_ = v.detach().requires_grad_(True)
        u_ = u.detach().requires_grad_(True) if u is not None else None
        with torch.enable_grad():
            f_ = rhs_fn(v_, u_)

        def f_at_v_only(z):
            u_det = u.detach() if u is not None else None
            with torch.enable_grad():
                return rhs_fn(z, u_det)

        def _jt_apply(y_flat_):
            y_view = y_flat_.view_as(v.detach())
            with torch.enable_grad():
                z = v.detach().clone().requires_grad_(True)
                fv = rhs_fn(z, u.detach() if u is not None else None)
            g = torch.autograd.grad(
                outputs=fv, inputs=z,
                grad_outputs=y_view, retain_graph=False, allow_unused=True,
            )[0]
            if g is None:
                return torch.zeros_like(y_flat_)
            return g.reshape(-1).detach()

        grad_flat = grad_out.contiguous().view(-1)
        if mode == 'phantom':
            # Phantom gradient (Geng et al. 2021): y <- grad_out - damp*(J^T y_prev - grad_out).
            # One step of this is the zeroth-order damped correction; multiple steps refine.
            y_flat = grad_flat.clone()
            for _ in range(max(phantom_steps, 1)):
                Jty = _jt_apply(y_flat)
                y_flat = grad_flat - phantom_damp * (Jty - grad_flat)
        else:
            y_flat = _solve_jt(f_at_v_only, v.detach(), grad_out,
                               tol=bt_tol, max_iter=bt_max).reshape(-1)
        y = (-y_flat).view_as(grad_out)

        if v_.grad is None:
            v_.grad = torch.zeros_like(v_)
        f_.backward(gradient=y, retain_graph=False)

        u_grad = u_.grad if u_ is not None else None
        return None, None, u_grad, None


class EquilibriumBlock(nn.Module):
    """Composed-mode DEQ block: solve f_k(v_k, v_{k-1}) = 0 per layer.

    Each CircuitLayer reaches its own equilibrium, receiving the previous
    layer's equilibrium v_{k-1} as its initial condition.  Only the first
    layer is driven by the input u.

    TODO(v2): add fused mode via topology.merge_topologies for the physical
              single-equilibrium case (all layers settle simultaneously).
    """

    def __init__(self, layer_list, solver_cfg, input_cfg):
        super(EquilibriumBlock, self).__init__()
        self.layer_list = nn.ModuleList(layer_list)
        self.solver_cfg = solver_cfg
        self.input_cfg = input_cfg or {}
        self.input_nodes = self.input_cfg.get('input_nodes', None)
        self.last_infos = []

    def prepare(self, device: List[int]) -> None:
        for layer in self.layer_list:
            layer.prepare(device)
            device_index = device[0] if device[0] >= 0 else None
            layer.set_input_nodes(self.input_nodes, device_index=device_index)

    def forward(self, u, v0=None, return_middle=False):
        middle = []
        self.last_infos = []
        last_v_star = None
        batch = u.shape[0]
        device = u.device
        dtype = u.dtype
        for i, layer in enumerate(self.layer_list):
            layer_u = u  # every layer sees u so all layer parameters receive gradients
            n = layer.max_node_index
            if (i == 0 and v0 is not None and v0.shape[-1] == n
                    and v0.shape[0] == batch):
                init = v0
            elif (last_v_star is not None and last_v_star.shape[-1] == n
                    and last_v_star.shape[0] == batch):
                init = last_v_star.detach()  # warm start; init enters no_grad anyway
            else:
                init = torch.zeros(batch, n, device=device, dtype=dtype)

            def rhs_fn(v, _u=layer_u, _layer=layer):
                return _layer.rhs(v, _u)

            v_star = EquilibriumSolve.apply(rhs_fn, init, layer_u, self.solver_cfg)
            last_v_star = v_star
            info = EquilibriumSolve.last_info
            self.last_infos.append({'layer': i,
                                     'n_iter': info.get('n_iter', -1) if info else -1,
                                     'final_residual': info.get('final_residual', float('inf')) if info else float('inf'),
                                     'converged': info.get('converged', False) if info else False})
            if return_middle:
                middle.append(v_star)
        return (last_v_star, middle)

    def solver_stats(self):
        return list(self.last_infos)


class LinearSolveLayer(nn.Module):
    """Equilibrium of f(w) = p - R w  <=>  w* = R^{-1} p.

    R must be symmetric positive definite.  Forward is solved by Anderson;
    backward uses the implicit-function theorem (one VJP + a fixed-point on
    the transpose operator, same SPD structure, monotone rate lambda_min(R)).

    TODO(v2): integrate into the main EquilibriumBlock as the second-order
              contribution for streaming RLS-style adaptive filtering demos.
              The spec's §6 design is preserved here in the docstring:

              # build R by accumulating xx^T over the stream
              # build p by accumulating e * x over the stream
              w_star = LinearSolveLayer()(p, R)         # analog Newton step
              e_next = d_next - w_star @ x_next

    Affine f means backward is exact and cheap (no iterative linear solve
    required — just one more equilibrium of the same operator).
    """

    def __init__(self, max_iter=50, tol=1e-6, beta=1.0):
        super().__init__()
        self.solver_cfg = {'method': 'anderson', 'max_iter': max_iter, 'tol': tol,
                            'backward_mode': 'exact', 'beta': beta}

    def _matvec_R(self, w, R):
        if R.dim() == 3:
            return torch.einsum('bij,bj->bi', R, w)
        # R is (n, n); w is (B, n).  Compute R w per batch row: (R @ w.T).T = w @ R.T.
        return w @ R.t()

    def rhs(self, w, p, R):
        return p - self._matvec_R(w, R)

    def forward(self, p, R):
        init = torch.zeros_like(p)
        last_v_star = None

        def rhs_fn(w, _p=p, _R=R):
            return self.rhs(w, _p, _R)

        v_star = EquilibriumSolve.apply(rhs_fn, init, p, self.solver_cfg)
        return v_star
