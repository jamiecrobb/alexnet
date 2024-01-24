from torch import nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        
        self.l1 = nn.Sequential(
            nn.Conv2d(3, 48, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(48, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(128, 192, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(192, 192, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.l5 = nn.Sequential(
            nn.Conv2d(192, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),  # Adjust the input size based on the new feature map size
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)

        x = x.view(-1, 128 * 4 * 4)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x