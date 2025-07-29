import torch
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler

def gen_data_mnist():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    dataset_train = datasets.FashionMNIST(
        "../data", train=True, download=True, transform=transform
    )
    dataset_test = datasets.FashionMNIST("../data", train=False, transform=transform)

    train_kwargs = {"batch_size": 256}
    test_kwargs = {"batch_size": 1024}

    kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": True}
    train_kwargs.update(kwargs)
    test_kwargs.update(kwargs)

    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    return train_loader, test_loader


def gen_data_cifar():
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            # (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            # (0.2023, 0.1994, 0.2010)),
        ]
    )

    dataset_train = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )

    dataset_test = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_kwargs = {"batch_size": 128}
    test_kwargs = {"batch_size": 100}

    kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": True}
    train_kwargs.update(kwargs)
    test_kwargs.update(kwargs)

    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    return train_loader, test_loader


def gen_data_imagenet(data_dir="/path/to/imagenet"):
    # Standard ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Training transform (resize -> crop -> flip -> normalize)
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # Validation transform (resize -> center crop -> normalize)
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # ImageNet folder structure:
    # train/ -> class folders
    # val/   -> class folders
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform_train)
    val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform_val)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256,
        sampler=train_sampler,
        num_workers=8, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256,
        sampler=val_sampler,
        num_workers=8, pin_memory=True
    )

    return train_loader, val_loader