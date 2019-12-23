import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np



class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),  # 3*32*32 -> 16*32*32
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, padding=1),  # 16*32*32 -> 32*32*32
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  # 32*32*32-> 32*16*16
            torch.nn.Conv2d(32, 64, 3, padding=1),  #  32*16*16 -> 64*16*16
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, padding=1),  #  64*16*16 -> 128*16*16
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  # 128*16*16 -> 128*8*8
            torch.nn.Conv2d(128, 256, 3, padding=1),  #  128*8*8 -> 256*8*8
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)  # 256*8*8 -> 256*4*4
        )

        self.gap = torch.nn.AvgPool2d(4,4)
        self.fc = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.gap(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


def train():
    for step, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)

        output = model(data)

        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    

def test():
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            images, labels = data.to(device), target.to(device)
            output = model(images)
            predicted = torch.max(output.data, 1)[1]
            correct += (predicted == labels).sum().item()

    print('Accuracy: {:.3f}%\n'.format(100. * correct / len(testloader.dataset)))


if __name__ == "__main__":
    LR = 0.005
    EPOCHS = 10
    BATCH_SIZE = 256
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, download=True,transform=tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    testloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    model = Model().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        print ('Epoch: {}/{}'.format(epoch+1, EPOCHS)) 
        train()
        test()