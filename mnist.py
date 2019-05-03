from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from blur import GaussianSmoothing
import random
import time
from datagen import generate, load
import sys

torch.manual_seed(1)
random.seed(0)

LR = 0.1
MOM = 0.5
HIDDEN = 50

class Net(nn.Module):
    def __init__(self, net_type):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, 10)
        self.net_type = net_type

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        if self.net_type == 'negative':
            x = x.neg()
        if self.net_type == 'negative_relu' or 'hybrid' in self.net_type:
            x = torch.ones_like(x).add(x.neg())
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                model.net_type, epoch, batch_idx * len(data), len(train_loader.dataset),
                                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('[{}] Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        model.net_type, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
kwargs = {'num_workers': 12, 'pin_memory': True} if use_cuda else {}

if '--generate' in sys.argv:
    generate()

train_loader = load('data/train_loader.pickle')
test_loader = load('data/test_loader.pickle')
test_loader_vertical_cut = load('data/test_loader_vcut.pickle')
test_loader_horizontal_cut = load('data/test_loader_hcut.pickle')
test_loader_diagonal_cut = load('data/test_loader_dcut.pickle')
test_loader_quarter_cut = load('data/test_loader_qcut.pickle')
test_loader_triple_cut = load('data/test_loader_tcut.pickle')
test_loader_triple_cut_noise = load('data/test_loader_noise.pickle')
test_loader_triple_cut_replaced1 = load('data/test_loader_replaced1.pickle')
test_loader_triple_cut_replaced3 = load('data/test_loader_replaced3.pickle')
test_loader_triple_cut_blur = load('data/test_loader_blur.pickle')


# import matplotlib.pyplot as plt

# plt.imshow(test_loader.dataset.test_data[343], cmap='gray')
# plt.show()
# plt.imshow(test_loader_vertical_cut.dataset.test_data[343], cmap='gray')
# plt.show()
# plt.imshow(test_loader_horizontal_cut.dataset.test_data[343], cmap='gray')
# plt.show()
# plt.imshow(test_loader_diagonal_cut.dataset.test_data[343], cmap='gray')
# plt.show()
# plt.imshow(test_loader_triple_cut.dataset.test_data[343], cmap='gray')
# plt.show()

# import sys
# sys.exit(0)

model_normal = Net('normal').to(device)
# model_negative = Net('negative').to(device)
model_negative_relu = Net('negative_relu').to(device)
model_hybrid = Net('normal').to(device)
model_hybrid_nr = Net('normal').to(device)
model_hybrid_alt = Net('normal').to(device)

optimizer_normal = optim.SGD(filter(lambda p: p.requires_grad, model_normal.parameters()), lr=LR, momentum=MOM)
# optimizer_negative = optim.SGD(filter(lambda p: p.requires_grad, model_negative.parameters()), lr=LR, momentum=MOM)
optimizer_negative_relu = optim.SGD(filter(lambda p: p.requires_grad, model_negative_relu.parameters()), lr=LR, momentum=MOM)
optimizer_hybrid = optim.SGD(filter(lambda p: p.requires_grad, model_hybrid.parameters()), lr=LR, momentum=MOM)
optimizer_hybrid_nr = optim.SGD(filter(lambda p: p.requires_grad, model_hybrid_nr.parameters()), lr=LR, momentum=MOM)
optimizer_hybrid_alt = optim.SGD(filter(lambda p: p.requires_grad, model_hybrid_alt.parameters()), lr=LR, momentum=MOM)

start_time = time.time()

for epoch in range(1, 10 + 1):
    train(model_normal, device, train_loader, optimizer_normal, epoch)

# for epoch in range(1, 10 + 1):
#     train(model_negative, device, train_loader, optimizer_negative, epoch)

for epoch in range(1, 10 + 1):
    train(model_negative_relu, device, train_loader, optimizer_negative_relu, epoch)

# ---- Hybrid net:

for epoch in range(1, 10 + 1):
    train(model_hybrid, device, train_loader, optimizer_hybrid, epoch)

# change network type
model_hybrid.net_type = 'hybrid'
# reinitialize fully connected layers
model_hybrid.fc1 = nn.Linear(320, HIDDEN).cuda()
model_hybrid.fc2 = nn.Linear(HIDDEN, 10).cuda()
# freeze convolutional layers
model_hybrid.conv1.weight.requires_grad = False
model_hybrid.conv2.weight.requires_grad = False
# reinitialize the optimizer with new params
optimizer_hybrid = optim.SGD(filter(lambda p: p.requires_grad, model_hybrid.parameters()), lr=LR, momentum=MOM)

for epoch in range(11, 20 + 1):
    train(model_hybrid, device, train_loader, optimizer_hybrid, epoch)

# ---- Hybrid no reset:

for epoch in range(1, 10 + 1):
    train(model_hybrid_nr, device, train_loader, optimizer_hybrid_nr, epoch)

# change network type
model_hybrid_nr.net_type = 'hybrid_nr'
# DO NOT reinitialize fully connected layers
# freeze convolutional layers
model_hybrid_nr.conv1.weight.requires_grad = False
model_hybrid_nr.conv2.weight.requires_grad = False
# reinitialize the optimizer with new params
optimizer_hybrid_nr = optim.SGD(filter(lambda p: p.requires_grad, model_hybrid_nr.parameters()), lr=LR, momentum=MOM)

for epoch in range(11, 20 + 1):
    train(model_hybrid_nr, device, train_loader, optimizer_hybrid_nr, epoch)

# ---- Hybrid alternating:

for epoch in range(1, 10 + 1):
    train(model_hybrid_alt, device, train_loader, optimizer_hybrid_alt, epoch)

# change network type
model_hybrid_alt.net_type = 'hybrid_alt'
# reinitialize fully connected layers
model_hybrid_alt.fc1 = nn.Linear(320, HIDDEN).cuda()
model_hybrid_alt.fc2 = nn.Linear(HIDDEN, 10).cuda()
# freeze convolutional layers
model_hybrid_alt.conv1.weight.requires_grad = False
model_hybrid_alt.conv2.weight.requires_grad = False
# reinitialize the optimizer with new params
optimizer_hybrid_alt = optim.SGD(filter(lambda p: p.requires_grad, model_hybrid_alt.parameters()), lr=LR, momentum=MOM)

for epoch in range(11, 20 + 1):
    if epoch % 2:
        model_hybrid_alt.net_type = 'normal'
    else:
        model_hybrid_alt.net_type = 'hybrid_alt'

    train(model_hybrid_alt, device, train_loader, optimizer_hybrid_alt, epoch)

# Testing:

models = [model_normal, model_negative_relu, model_hybrid, model_hybrid_nr, model_hybrid_alt]

datasets = [test_loader, test_loader_horizontal_cut, test_loader_vertical_cut, test_loader_diagonal_cut, test_loader_quarter_cut]
dataset_names = ['Normal:', 'HCUT:', 'VCUT:', 'DCUT:', 'QCUT:']

for i in range(3):
    size = "{0}x{0}".format(5 + 2 * i)
    datasets.append(test_loader_triple_cut[i])
    dataset_names.append("TCUT {}:".format(size))
    datasets.append(test_loader_triple_cut_blur[i])
    dataset_names.append("Blur {}:".format(size))
    datasets.append(test_loader_triple_cut_noise[i])
    dataset_names.append("Noise {}:".format(size))
    datasets.append(test_loader_triple_cut_replaced1[i])
    dataset_names.append("Replaced1 {}:".format(size))
    datasets.append(test_loader_triple_cut_replaced3[i])
    dataset_names.append("Replaced3 {}:".format(size))

for i, dataset in enumerate(datasets):
    print('Testing -- ' + dataset_names[i])
    for model in models:
        test(model, device, dataset)

print('--- Total time: %s seconds ---' % (time.time() - start_time))

torch.save(model_normal, 'models/model_normal.pytorch')
# torch.save(model_negative, 'models/model_negative.pytorch')
torch.save(model_negative_relu, 'models/model_negative_relu.pytorch')
torch.save(model_hybrid, 'models/model_hybrid.pytorch')
torch.save(model_hybrid_nr, 'models/model_hybrid_nr.pytorch')
torch.save(model_hybrid_alt, 'models/model_hybrid_alt.pytorch')

print('models saved to "models"')
