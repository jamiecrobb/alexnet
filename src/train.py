import torch
from torch import optim

import os
from setup import project_root

from src.model import AlexNet
from data import get_cifar10_loaders


def train_model(model, train_loader, criterion, optimizer, device, weights_path, num_epochs=1, print_every=100):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU

            optimizer.zero_grad()

            outputs = model(inputs)
            labels = labels.squeeze()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % print_every == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / print_every))
                running_loss = 0.0
        
        model_path = f'{weights_path}/alexnet_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), model_path)
        print(f'Model saved at epoch {epoch+1}.')
    print('Finished Training')


def main():
    weights_path = os.path.join(project_root, 'weights')

    alexnet = AlexNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimiser = optim.SGD(alexnet.parameters(), momentum=0.9, lr=0.005, weight_decay=0.005)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alexnet.to(device)

    train_loader, test_loader = get_cifar10_loaders()

    train_model(model=alexnet, 
                train_loader=train_loader, 
                criterion=criterion, 
                optimizer=optimiser, 
                device=device,
                weights_path=weights_path)
    
if __name__ == "__main__":
    main()
