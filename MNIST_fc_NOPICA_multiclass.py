import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.autograd import Variable
from Fullyconnected_nets import LogisticR, FC
import pickle

import torchvision
import torchvision.datasets as datasets

torch.manual_seed(0)

trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_set = datasets.MNIST(root='./data', train=True, transform=trans, download=True)
test_set = datasets.MNIST(root='./data', train=False, transform=trans, download=True)

epoch = 300
batch_size = 200
lr1 = .001

S = []
train_acc = [0] * epoch
train_loss = [0] * epoch


for exper in range(10):
    print('exper: ', exper)
    fc = 1
    if not fc:
        model = LogisticR(784, 10)
    else:
        model = FC(784, 10)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr1)
    criterion = torch.nn.CrossEntropyLoss()

    for ep in range(epoch):
        correct = 0.0
        total = 0.0
        for i, (samples, y) in enumerate(train_loader):
            samples = Variable(samples).view(-1, 784)
            samples = samples.float()
            samples = samples / torch.sum(samples, dim=-1).view(-1, 1)
            y = Variable(y).view(-1)
            output = model(samples)
            predicts = (output.data.max(dim=-1)[1])
            loss = criterion(output, y)
            loss.backward(retain_graph=True)
            optimizer.step()
            total += y.size(0)
            correct += (predicts == y).sum().item()
            train_loss[ep] += loss.data.item()
        train_acc[ep] = correct / total * 100.
        # if ep % 10 == 0:
        print(ep, correct / total * 100.)
    correct = 0.0
    total = 0.0
    for j, (samples_, y_) in enumerate(test_loader):
        samples_ = Variable(samples_).view(-1, 784)
        samples_ = samples_.float()
        samples_ = samples_ / torch.sum(samples_, dim=1).view(-1, 1)
        y_ = Variable(y_).view(-1)
        output = model(samples_)
        total += y_.size(0)
        predicts = (output.data.max(dim=-1)[1])
        correct += (predicts == y_).sum().item()
    acc = correct / total * 100.
    S.append(acc)
    print(ep, acc)

print('---')
print(S, sum(S) / len(S))

train_acc = [ta / 10. for ta in train_acc]
with open("train_loss_lr.txt", "wb") as fp:  # Pickling
    pickle.dump([train_acc, train_loss], fp)






