import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils.testbench import generate_testbench
from utils.model import CircuitNet
from utils.topology import generate_topology
from utils.utils import update_metric, parse_scheduler, set_seed
from utils.deq_solver import check_contraction
import os
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='deq_debug')
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--gpu', type=int, nargs='+', default=[-1])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./results')
    return parser.parse_args()


def pre_experiment(args):
    device = torch.device(
        f"cuda:{args.gpu[0]}" if torch.cuda.is_available() and args.gpu[0] >= 0 else 'cpu'
    )
    set_seed(args.seed)
    exp_dir = os.path.join(args.save_dir, args.exp_name + '_seed_' + str(args.seed))
    if not os.path.exists(exp_dir):
        try:
            os.mkdir(exp_dir)
        except OSError:
            pass
    else:
        for file in os.listdir(exp_dir):
            try:
                os.remove(os.path.join(exp_dir, file))
            except OSError:
                pass
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    return device, config, exp_dir


def contract_check(model, device):
    """Run the analytic contraction certificate on the first layer.

    Returns lambda_max(J_hat) and a flag for whether passivity holds.
    """
    layer = model.circuit.layer_list[0]
    src = layer.src_indices_list[0] if layer.src_indices_list[0] is not None else layer.src_node
    des = layer.des_indices_list[0] if layer.des_indices_list[0] is not None else layer.des_node
    src = src.to(device)
    des = des.to(device)
    n = layer.max_node_index
    D = layer.worst_case_D().to(device).detach()
    if D.dim() == 0:
        D = D.expand(layer.num_edge)
    gamma_diag = layer.gamma.detach().expand(n).clone()
    return check_contraction(src, des, n, D, gamma_diag)


def main():
    args = parse_args()
    device, config, exp_dir = pre_experiment(args)
    dp_flag = args.gpu[0] >= 0 and len(args.gpu) > 1

    solver_cfg = config.get('solver', {'method': 'anderson',
                                         'max_iter': 100,
                                         'tol': 1e-6})
    input_cfg = config.get('fabric', {'input_nodes': [1]})

    model = CircuitNet(
        circuit_topology=[generate_topology(ns) for ns in config['network']['net_struct']],
        sim_dict=config.get('simulation', {'t_end': [0.0, 1.0],
                                              'tol': 1e-3, 'min_step': 1e-5,
                                              'first_step': 1e-3, 'step_size': 1e-6,
                                              'method': 'dopri5'}),
        circuit_dict={'model': config['network']['model'],
                       'initialization': config['network']['initialization'],
                       'residual': config['network'].get('residual', None),
                       'fill': config['network'].get('fill', None)},
        encoder=config['network'].get('encoder', None),
        projector=config['network'].get('projector', None),
        use_augment=False,
        adjoint=False,
        mode=config.get('mode', 'deq'),
        solver_cfg=solver_cfg,
        input_cfg=input_cfg,
    ).to(device)

    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"### Mode: {model.mode}")
    print(f"### Test bench: {config['test_bench']}")
    print(f"### Seed: {args.seed}")
    print(f"### Net: {config['network']['net_struct']}")
    print(f"### Params: {num_param:.6f} MiB")
    for i, L in enumerate(model.circuit.layer_list):
        print(f"### Layer {i}: edges={L.num_edge}, max_node={L.max_node_index}, gamma={L.gamma.item():.4f}")
    print(f"### Solver cfg: {solver_cfg}")
    print(f"### Input cfg: {input_cfg}")

    train_loader, test_loader, criterion, process_func = generate_testbench(
        config['test_bench'],
        config['batch_size'] * max(1, len(args.gpu)),
        config['num_worker'],
    )

    optimizer = optim.AdamW(model.parameters(), lr=float(config['train']['lr']))
    scheduler = parse_scheduler(config['train'].get('scheduler'), optimizer)

    model.prepare(args.gpu)
    if dp_flag:
        model = nn.DataParallel(model, device_ids=args.gpu)

    loss_list, stats_list = [], []
    print("Begin training")
    contract_check_every = config.get('contract_check_every', 5)
    for epoch in range(1, 1 + config['train']['num_epoch']):
        running_loss, total_num, ep_start = 0.0, 0, time.time()
        for i, data in enumerate(train_loader):
            inputs, labels = process_func(data)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.shape[0]
            total_num += inputs.shape[0]
            print(f"E{epoch:03d} it{i:04d} loss={loss.item():.3e}")
        running_loss /= total_num
        ep_end = time.time()
        print(f"=== Epoch {epoch}/{config['train']['num_epoch']} ({ (ep_end-ep_start)/60:.2f}m) train_loss={running_loss:.3e}")

        loss_list.append(running_loss)
        stats = model.solver_stats() if hasattr(model, 'solver_stats') else None
        if stats is None and hasattr(model, 'module'):
            stats = model.module.circuit.solver_stats()
        elif stats is None:
            stats = model.circuit.solver_stats()
        stats_list.append(stats)
        print(f"=== Solver stats: {stats}")

        if epoch % contract_check_every == 0 or epoch == config['train']['num_epoch']:
            try:
                cp = contract_check(model.module if dp_flag else model, device)
                print(f"=== Contraction: lambda_max(J)={cp['lambda_max_J']:.4e}, "
                      f"lambda_min(M)={cp['lambda_min_M']:.4e}, "
                      f"lambda_max(M)={cp['lambda_max_M']:.4e}, "
                      f"margin={cp['contraction_margin']:.4e}")
                if not cp['passive']:
                    print("!!! WARNING: parameterization leaked passivity (lambda_max >= 0)")
            except Exception as e:
                print(f"!!! Certificate check failed: {e}")

        if epoch % config['train'].get('save_epoch', 1) == 0 or epoch == config['train']['num_epoch']:
            state = model.module.state_dict() if dp_flag else model.state_dict()
            torch.save(state, os.path.join(exp_dir, 'model.pth'))
            plt.figure()
            plt.plot(np.arange(len(loss_list)), np.array(loss_list))
            plt.savefig(os.path.join(exp_dir, 'train_loss.png'))
            plt.close()

        if scheduler is not None:
            scheduler.step()

    print("Begin testing")
    model.eval()
    metric_value = {key: 0.0 for key in config['test'].get('metric', [])}
    test_loss = 0.0
    total_num = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = process_func(data)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.shape[0]
            total_num += inputs.shape[0]
            metric_value = update_metric(metric_value, outputs, labels)
    test_loss /= max(1, total_num)
    for k in metric_value:
        metric_value[k] /= max(1, total_num)
    print(f"=== Test loss={test_loss:.3e}, metrics={metric_value}")

    metric_value['test_loss'] = test_loss
    metric_value['num_param_mib'] = num_param
    metric_value['train_loss_list'] = loss_list
    metric_value['solver_stats_list'] = stats_list
    with open(os.path.join(exp_dir, 'metric.yaml'), 'w') as f:
        yaml.dump(metric_value, f)


if __name__ == '__main__':
    main()
