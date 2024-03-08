import os
from tqdm import tqdm

from data import Data
from model import Model

import torch
import torch.optim as optim


class Trainer:
    def __init__(self, data_path, weights_path, num_epochs: int = 1):

        self.data_handler = Data(data_path)
        self.train_loader, _, _ = self.data_handler.get_loaders()

        # Ensure the weights directory definitely exists
        self.weights_dir = os.path.expanduser(weights_path)
        os.makedirs(self.weights_dir, exist_ok=True)

        # Get model and send to appropriate device
        self.model = Model()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Configure training parameters
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimiser = optim.SGD(self.model.parameters(
        ), momentum=0.9, lr=0.005, weight_decay=0.005)
        self.num_epochs = num_epochs

    def train(self):
        """
        TODO: docstring and some comments throughout this would be good
        """

        print('Starting training...')
        for epoch in tqdm(range(self.num_epochs)):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimiser.zero_grad()

                outputs = self.model(inputs)
                labels = labels.squeeze()

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimiser.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(self.train_loader)
            print(f"Loss at epoch {epoch+1}: {epoch_loss:.3f}")
            running_loss = 0.0

            model_path = os.path.join(
                self.weights_dir, f'alexnet_epoch_{epoch}.pth')
            torch.save(self.model.state_dict(), model_path)

        print('Finished Training')


def main():

    project_root = "~/code/personal/alexnet"
    data_path = project_root + "/data"
    weights_path = project_root + "/weights"

    trainer = Trainer(data_path=data_path,
                      weights_path=weights_path,
                      num_epochs=10)
    trainer.train()


if __name__ == "__main__":
    main()
