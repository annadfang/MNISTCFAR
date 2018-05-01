
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainloader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10(root='./data', train=True, download = True, transform=transform), 
    batch_size=64, shuffle=True, num_workers=1)

testloader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10(root='./data', train=False, download = True, transform=transform), 
    batch_size=64, shuffle=True, num_workers=1)

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim = 1)

model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate (trainloader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad() #clear gradients
        # each time variable is back propagated, the gradient will accumulate instead of replaced
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx % 10 == 0):
            print('Training Epoch: ', epoch, '  Loss: ', loss.data[0])
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data,target in testloader:
        data, target = Variable(data, volatile = True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] 
        pred = output.data.max(1, keepdim=True)[1] 
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))


for epoch in range(10):
    train(epoch)
    test()
