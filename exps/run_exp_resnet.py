import os
import pickle

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import DistributedSampler

from exps.qat_config_fp import model_name, qconfigs, save_name
from quant_mp.data_gen import gen_data_cifar, gen_data_mnist
from quant_mp.models import ConvNet, LinNet, ResNet18
from quant_mp.train import test, train


def init_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def model_select(name, qconfig):
    if name == "LinNet":
        lr = 0.001
        gamma = 0.7
        epochs = 10
        model = LinNet(qconfig)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
        gen_dataset = gen_data_mnist

    elif name == "ConvNet":
        lr = 0.001
        gamma = 0.7
        epochs = 10
        model = ConvNet(qconfig)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
        gen_dataset = gen_data_cifar

    elif name == "ResNet":
        lr = 0.1
        epochs = 40
        model = ResNet18(qconfig)
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        gen_dataset = gen_data_mnist

    else:
        raise ValueError(f"Unknown model name: {name}")

    return model, optimizer, scheduler, epochs, gen_dataset


def main():
    local_rank = init_distributed()
    rank = dist.get_rank()

    qconfig = qconfigs[0]

    model, optimizer, scheduler, epochs, gen_dataset = model_select(model_name, qconfig)
    device = torch.device(f"cuda:{local_rank}")
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    train_loader, test_loader = gen_dataset()
    train_loader.sampler = DistributedSampler(train_loader.dataset)
    test_loader.sampler = DistributedSampler(test_loader.dataset, shuffle=False)

    loss_vec = []
    loss_vec_test = []
    for epoch in range(1, epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        loss_vec += train(model, device, train_loader, optimizer, epoch)
        loss_vec_test += test(model, device, test_loader)
        scheduler.step()

    if rank == 0:
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        with open(save_name, "wb") as handle:
            pickle.dump(
                {"loss": loss_vec, "loss_test": loss_vec_test, "qconfig": qconfig},
                handle,
            )
        print(f"Results saved to {save_name}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
