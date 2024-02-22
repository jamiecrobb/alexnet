from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import os

def get_cifar10_loaders(batch_size=16, data_path=None):
    """
    Function to download CIFAR10 datasets and return train- and test-loaders.
    """

    if data_path is None:
        data_path = os.path.join(project_root, 'data')
    else:
        data_path = os.path.abspath(data_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32)),
        transforms.CenterCrop((32, 32)),
    ])

    train_dataset = CIFAR10(root=data_path, train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root=data_path, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
