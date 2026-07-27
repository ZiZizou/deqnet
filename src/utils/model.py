import torch
import torch.nn as nn
from .circuit_block import CircuitBlock, CircuitLayer, AugCircuitLayer, AugCircuitBlock, EquilibriumBlock
from typing import List, Dict, Union, Optional, Any, Tuple
from .baseline import FeedForwardNet
import re
import numpy as np

class LastNet(nn.Module):
    def __init__(self, num):
        super(LastNet,self).__init__()
        self.num = num
    def forward(self,x):
        return x[..., -self.num:]
class FirstNet(nn.Module):
    def __init__(self, num):
        super(FirstNet,self).__init__()
        self.num = num
    def forward(self,x):
        return x[..., :self.num]
class ReshapeNet(nn.Module):
    def __init__(self, node_feature):
        super(ReshapeNet, self).__init__()
        self.node_feature = node_feature
    def forward(self,x):
        return x.view(self.node_feature, x.shape[1] // self.node_feature).t()

def generate_projector(keyword: Union[List, str]) -> Union[nn.Module, callable, None]:
    if keyword == 'none' or keyword is None:
        model = None
    elif re.match(r'^mlp_(\d+)_(\d+)$', keyword):
        num1 = int(re.match(r'^mlp_(\d+)_(\d+)$', keyword).group(1))
        num2 = int(re.match(r'^mlp_(\d+)_(\d+)$', keyword).group(2))
        print(f"matched num1: {num1}, num2: {num2}")
        model = FeedForwardNet([num1, num2])

    elif re.match(r'^last(\d+)$', keyword):
        # Extract the number from the keyword using regular expression, e.g. keyword = 'last10' gives num = 10
        num = int(re.match(r'^last(\d+)$', keyword).group(1))
        model = LastNet(num)
    elif re.match(r'^first(\d+)$', keyword):
        # Extract the number from the keyword using regular expression, e.g. keyword = 'last10' gives num = 10
        num = int(re.match(r'^first(\d+)$', keyword).group(1))
        model = FirstNet(num)
    elif re.match(r'^reshape(\d+)$', keyword):
        num_node_feature = int(re.match(r'^reshape(\d+)$', keyword).group(1))
        model = ReshapeNet(num_node_feature)
    else:
        raise NotImplementedError(f"Unsupported keyword: {keyword}")
    return model


def generate_encoder(keyword: Union[str, List, None]) -> Union[nn.Module, callable, None]:
    if keyword == 'none' or keyword is None:
        model = None
    elif isinstance(keyword, list):
        model = FeedForwardNet(keyword)
    elif re.match(r'^mlp_(\d+)_(\d+)$', keyword):
        m = re.match(r'^mlp_(\d+)_(\d+)$', keyword)
        num1 = int(m.group(1))
        num2 = int(m.group(2))
        model = FeedForwardNet([num1, num2])
    else:
        raise NotImplementedError(f"Unsupported encoder keyword: {keyword}")
    return model


class CircuitNet(nn.Module):
    def __init__(self, circuit_topology, sim_dict: Dict[str, Any], circuit_dict: Dict[str, Any],
                 encoder: Optional[str] = None,
                 projector: Optional[str] = None, use_augment: Optional[bool] = False,
                 adjoint: Optional[bool] = True,
                 mode: Optional[str] = 'ode',
                 solver_cfg: Optional[Dict[str, Any]] = None,
                 input_cfg: Optional[Dict[str, Any]] = None):
        super(CircuitNet, self).__init__()
        self.mode = mode
        self.use_augment = use_augment
        self.input_cfg = input_cfg or {}
        self.solver_cfg = solver_cfg or {'method': 'anderson',
                                          'max_iter': 100,
                                          'tol': 1e-6}

        if mode == 'ode':
            from torchdiffeq import odeint, odeint_adjoint
            if not use_augment:
                circuit_layer_list = [CircuitLayer(topo, circuit_dict) for topo in circuit_topology]
                self.circuit = CircuitBlock(circuit_layer_list, sim_dict, circuit_dict['residual'],
                                            odeint_adjoint if adjoint else odeint, circuit_dict['fill'])
            else:
                circuit_layer_list = [AugCircuitLayer(topo, circuit_dict) for topo in circuit_topology]
                self.circuit = AugCircuitBlock(circuit_layer_list, sim_dict, circuit_dict['residual'],
                                               odeint_adjoint if adjoint else odeint, circuit_dict['fill'])
        elif mode == 'deq':
            circuit_layer_list = [CircuitLayer(topo, circuit_dict) for topo in circuit_topology]
            self.circuit = EquilibriumBlock(circuit_layer_list, self.solver_cfg, self.input_cfg)
        else:
            raise ValueError(f"Unknown mode: {mode!r}; expected 'ode' or 'deq'")

        self.encoder = generate_encoder(encoder)
        if isinstance(projector, list):
            self.projector = nn.ModuleList([generate_projector(proj) for proj in projector])
        else:
            self.projector = generate_projector(projector)

    def prepare(self, device: List[int]) -> None:
        if self.mode == 'deq':
            self.circuit.prepare(device)
        else:
            for layer in self.circuit.layer_list:
                layer.prepare(device)

    def forward(self, x: Union[torch.Tensor, tuple], reverse=False, return_middle=False) -> torch.Tensor:

        if self.mode == 'deq':
            u = self.encoder(x) if self.encoder is not None else x
            # Ensure u carries requires_grad so the autograd.Function machinery
            # hooks up the implicit-differentiation backward chain.
            if not u.requires_grad:
                u = u.detach().requires_grad_(True)
            x, middle = self.circuit(u, return_middle=return_middle)
        else:
            if self.encoder is not None:
                x = self.encoder(x)
            x, middle = self.circuit(x, reverse, return_middle)

        if isinstance(self.projector, nn.ModuleList):
            for prj in self.projector:
                x = prj(x)
        elif self.projector is not None:
            x = self.projector(x)

        return (x, middle)

    @property
    def nfe(self):
        if self.mode == 'deq':
            stats = self.circuit.solver_stats()
            return [{'n_iter': s['n_iter'], 'residual': s['final_residual']} for s in stats]
        return [layer.nfe.item() for layer in self.circuit.layer_list]

    @nfe.setter
    def nfe(self, value: Union[List, Tuple, np.ndarray]):
        if self.mode == 'ode':
            for i in range(len(self.circuit.layer_list)):
                self.circuit.layer_list[i].nfe = value[i]

    def solver_stats(self):
        if self.mode == 'deq':
            return self.circuit.solver_stats()
        return None
