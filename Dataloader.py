import torch
from torch.utils.data import DataLoader as dataloader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

transformer = transforms.Compose([

    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def get_train_data():
    train_data = datasets.MNIST(root='./data',
                                download=True,
                                train=True,
                                transform=transformer)

    train_loader = dataloader(train_data,
                              batch_size=64,
                              shuffle=True)

    return train_data, train_loader


def test_val_data():
    t_v_data = datasets.MNIST(root='./data',
                              download=True,
                              train=False,
                              transform=transformer)

    indices = range(len(t_v_data))
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:5000])
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[5000:])

    test_dataloader = dataloader(t_v_data, sampler=test_sampler, batch_size=64)
    val_dataloader = dataloader(t_v_data, sampler=val_sampler, batch_size=64)

    return t_v_data, test_dataloader, val_dataloader


t_v_data, test_dataloader, val_dataloader = test_val_data()
