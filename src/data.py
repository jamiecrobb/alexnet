from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10


class Data:
    """
    A class to manage loading, preprocessing, and splitting of CIFAR-10 dataset.

    Attributes:
        data_path (str)
        split (tuple[float, float, float])

    Methods:
        _get_data()
        _create_datasplits()
        get_loaders(batch_size)
    """

    def __init__(self, data_path: str, split: tuple[float, float, float] = (0.6, 0.2, 0.2)):

        self.data_path = data_path
        self.split = split

        # Download the data and create splits
        self.dataset = self._get_data()
        self.train_set, self.val_set, self.test_set = self._create_datasplits()

    def _get_data(self) -> CIFAR10:
        """
        Define and apply the transform
        to get the master dataset
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32)),
            transforms.CenterCrop((32, 32)),
        ])

        return CIFAR10(root=self.data_path, train=True, download=True, transform=transform)

    def _create_datasplits(self) -> tuple[CIFAR10, CIFAR10, CIFAR10]:
        """
        Using the provided split ratios, randomly 
        split the data into 3 datasplits.
        Calculate the appropriate split sizes.
        """

        data_len = len(self.dataset)

        train_len = int(data_len * self.split[0])
        val_len = int(data_len * self.split[1])
        test_len = data_len - (train_len + val_len)

        train, val_test = random_split(
            self.dataset, [train_len, val_len+test_len])
        val, test = random_split(val_test, [val_len, test_len])

        return (train, val, test)

    def get_loaders(self, batch_size: int = 16) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Returns 3 dataloaders
        """

        train_loader = DataLoader(
            self.train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            self.val_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(
            self.test_set, batch_size=batch_size, shuffle=False)

        return (train_loader, val_loader, test_loader)
