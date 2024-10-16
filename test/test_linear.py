import torch
from quant_mp.models import LinNet, ConvNet, ResNet18
from quant_mp.train import train, test

import torch.optim as optim

from quant_mp.data_gen import gen_data_mnist, gen_data_cifar

from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import time

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os

qconfig =  {
        'label': "Float4-minmax-e2m1",
        'activation': {'qtype': 'uniform', 'qbits': 4, 'qblock_size': 'channel', 'alg': 'iterative', 'beta':0., 'format': 'e2m1'},
        'weight': {'qtype': 'uniform', 'qbits': 4, 'qblock_size': 'channel', 'alg': 'iterative', 'beta':0., 'format': 'e2m1'},
        'grad': {'qtype': None, 'qbits': 4, 'qblock_size': 'channel', 'alg': 'normal', 'beta':0., 'format': 'e2m1'}
    }

# qconfig =  {
#         'label': "FP32",
#         'activation': {'qtype': None},
#         'weight': {'qtype': None},
#         'grad': {'qtype': None}
#     }

lr = 0.0001
gamma = 0.7
epochs = 50
model = LinNet(qconfig)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)


def run(rank, world_size, qconfig, return_dict):

    print('Train on: ', rank)
    device = torch.device("cuda:{}".format(rank))
    train_loader, test_loader = gen_data_mnist()
    
    model.to(device)

    loss_vec = []
    loss_vec_test = []
    s_vec = []
    for epoch in range(1, epochs + 1):
        loss_vec += train(model, device, train_loader, optimizer, epoch)
        loss_vec_test += test(model, device, test_loader)
        scheduler.step()

    return_dict[qconfig['label']] = (loss_vec, loss_vec_test, s_vec)

    
if __name__ == "__main__":

    return_dict = {}
    run(0, 1, qconfig, return_dict)
    