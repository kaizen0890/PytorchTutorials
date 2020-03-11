# --------------------------------------------------------
# checkpoint example with pytorch
# Written by Huy Thanh Nguyen (kaizen0890@gmail.com)
# github:
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Download data and create Dataloader

train_loader = DataLoader(datasets.MNIST('./data',train=True,
                                         download=True,
                                         transform=transforms.Compose([transforms.ToTensor(),
                                                                       transforms.Normalize((0.1307,), (0.3081,))])),
                                         batch_size=8,shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False,
                                                         transform=transforms.Compose([
                                                             transforms.ToTensor(),
                                                             transforms.Normalize((0.1307,), (0.3081,))])),
                                          batch_size=8,shuffle=True)


# Define a CNN network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# train function
def train(epoch, log_interval=10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    # 1. Save the model every epoch
    torch.save(model.state_dict(), "mnist_model_{0:03d}.pth.tar".format(epoch))


# test function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# create the model
model = Net()
model = model.to(device)
# create the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

if __name__ == "__main__":
    # 2. Reload the model if asked
    epochs = 10
    resume_checkpoint = "./snapshot/checkpoint_e1.pth"
    beginEpoch = 2


    if resume_checkpoint:
        model.load_state_dict(torch.load(resume_checkpoint))

    for epoch in range(beginEpoch, epochs + 1):
        train(epoch)

    test()

