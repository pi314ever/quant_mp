import os
import pickle

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from qat_config import model_name, qconfigs, save_name
from torch.optim.lr_scheduler import StepLR

from quant_mp.data_gen import gen_data_cifar, gen_data_mnist
from quant_mp.models import ConvNet, LinNet, ResNet18
from quant_mp.train import test, train

# FIXME: Update to new architecture


def model_select(name, qconfig):
    if name == "LinNet":
        lr = 0.001
        gamma = 0.7
        epochs = 10
        model = LinNet(qconfig)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
        gen_dataset = gen_data_mnist

    if name == "ConvNet":
        lr = 0.001
        gamma = 0.7
        epochs = 10
        model = ConvNet(qconfig)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
        gen_dataset = gen_data_cifar

    if name == "ResNet":
        lr = 0.1
        gamma = 0.7
        epochs = 40
        model = ResNet18(qconfig)
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        gen_dataset = gen_data_cifar

    return model, optimizer, scheduler, epochs, gen_dataset


def run(rank, world_size, qconfig, return_dict):
    print("Train on: ", rank)
    device = torch.device("cuda:{}".format(rank))

    model, optimizer, scheduler, epochs, gen_dataset = model_select(model_name, qconfig)
    model.to(device)

    train_loader, test_loader = gen_dataset()

    loss_vec = []
    loss_vec_test = []
    s_vec = []
    for epoch in range(1, epochs + 1):
        loss_vec += train(model, device, train_loader, optimizer, epoch)
        loss_vec_test += test(model, device, test_loader)
        scheduler.step()

    if qconfig.weight:
        return_dict[
            (str(qconfig.weight.qval_data_format), str(qconfig.weight.algorithm))
        ] = (loss_vec, loss_vec_test, s_vec, qconfig)
    else:
        return_dict[("fp32", None)] = (loss_vec, loss_vec_test, s_vec, qconfig)


def init_process(rank, size, qconfig, return_dict, fn, backend="nccl"):
    """Initialize the distributed environment."""
    print("Initializing with size:", size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, qconfig, return_dict)


if __name__ == "__main__":
    # world_size  = torch.cuda.device_count()
    world_size = len(qconfigs)
    print("GPU: ", world_size)

    processes = []
    mp.set_start_method("spawn")
    manager = mp.Manager()
    return_dict = manager.dict()

    for rank, qconfig in enumerate(qconfigs):
        p = mp.Process(
            target=init_process, args=(rank, world_size, qconfig, return_dict, run)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    with open(save_name, "wb") as handle:
        pickle.dump(dict(return_dict), handle)
