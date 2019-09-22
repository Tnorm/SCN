import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.autograd import Variable
from SCN import SCN_multi, SCN_multi_justified
import pickle

import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import sys

torch.manual_seed(0)

trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_set = datasets.MNIST(root='./data', train=True, transform=trans, download=True)
test_set = datasets.MNIST(root='./data', train=False, transform=trans, download=True)


#print(train_data.shape, test_data.shape, train_targets.shape, test_targets.shape)

epoch = 100
batch_size = 200
lr1 = 0.5
scn_depth = 1

S = []
train_acc = [0] * epoch
train_loss = [0] * epoch



for exper in range(10):
    print('exper: ', exper)
    zero = Variable(torch.zeros(1, 783))
    eye = Variable(torch.eye(783))
    visible_units = torch.cat((zero, eye), 0)
    visible_units = visible_units.float()

    scn = SCN_multi_justified(784, 783, 10, visible_units, scn_depth, model=1)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(scn.parameters(), lr=lr1)
    criterion = torch.nn.CrossEntropyLoss()

    lamb = 1.0
    torch.autograd.set_detect_anomaly(True)

    for ep in range(epoch):
        correct = 0.0
        total = 0.0
        for i, (samples, y) in enumerate(train_loader):
            samples = Variable(samples).view(-1, 784)
            samples = samples.float() + torch.rand_like(samples) * 1e-10
            samples = samples / torch.sum(samples, dim=-1).view(-1, 1)
            #one_hot_y = torch.zeros(samples.shape[0], 10)
            y = Variable(y).view(-1)
            #one_hot_y = one_hot_y.scatter(1, y, 1)
            #y=y.float()
            output, _, last_h = scn(samples)
            output = output.view(-1, 10)
            predicts = (output.data.max(dim=-1)[1])
            loss = criterion(output, y)# + lamb * torch.norm(last_h - samples, dim=1).mean()
            #print(criterion(output, y), lamb * torch.norm(last_h - samples, dim=1).mean())
            loss.backward(retain_graph=True)
            # if i % 50 != 0:
            #     scn.L[0].grad.data.fill_(0.0)
            #print(scn.L[0].grad.data.max(), scn.L[0].grad.data.min())
            optimizer.step()
            #scn.bias_funcs[0].weight.data.fill_(0.0)
            volatility = 1

            #### if no softmax used, then use the following:
            # for j in range(scn.depth):
            #     #scn.L[j].data = (scn.L[j].data - lr1 * volatility * scn.L[j].grad.data).clamp(0.1, 10.)
            #     scn.L[j].data = scn.L[j].data.clamp(0.001, 0.2)
            #     scn.L[j].data = (scn.L[j].data / (scn.L[j].data.sum())).clamp(0., 1.)
            #     volatility *= 1.0
            ####

            # scn.biases.data = torch.zeros(scn.biases.data.size())
            #scn.biases.data = scn.biases.data.clamp(-100, 100)
            #print(scn.L[0].data.max(), scn.L[0].data.min(), scn.bias_funcs[0].weight.data)
            total += y.size(0)
            correct += (predicts == y).sum().item()
            train_loss[ep] += loss.data.item()
        train_acc[ep] += correct / total * 100.
        #if ep % 2 == 0:
        print(ep, correct / total * 100.)
        print([(scn.L[i].data.max(), scn.L[i].data.min()) for i in range(scn_depth)])
        print([(scn.bias_funcs[i].weight.max(), scn.bias_funcs[i].weight.min()) for i in range(scn_depth)])
        print(scn.biases[0])
        #print(scn.biases.data)
        #h11 = torch.matmul(scn.L[0].data, scn.visible_units)
        #print(h11.min(), h11.max())
    correct = 0.0
    total = 0.0
    for j, (samples_, y_) in enumerate(test_loader):
        samples_ = Variable(samples_).view(-1, 784)
        samples_ = samples_.float()
        samples_ = samples_ / torch.sum(samples_, dim=1).view(-1, 1)
        y_ = Variable(y_).view(-1)
        output = scn(samples_)[0].view(-1, 1)
        total += y_.size(0)
        predicts = (output.data.max(dim=-1)[1])
        correct += (predicts == y_).sum().item()
    acc = correct / total * 100.
    S.append(acc)
    print(ep, acc)

print('---')
print(S, sum(S) / len(S))

train_acc = [ta / 10. for ta in train_acc]
with open("train_loss_scn.txt", "wb") as fp:  # Pickling
    pickle.dump([train_acc, train_loss], fp)

