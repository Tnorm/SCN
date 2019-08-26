import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F

class FC(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(FC, self).__init__()
        # depth = number of hidden units
        self.fc1 = torch.nn.Linear(input_dim, 300)
        self.rlu = torch.nn.ReLU()
        self.sgmd = torch.nn.Sigmoid()
        #self.fc2 = torch.nn.Linear(300, 2)
        self.fc2 = torch.nn.Linear(300, 50)
        self.fc3 = torch.nn.Linear(50, output_dim)
        #self.fc1.data = torch.zeros(input_dim, 300)
        #self.fc2.data = torch.zeros(300, 1)
        #self.fc3.data = torch.zeros(300,1)
        ## for one dimensional data
        #self.fc1.bias.data = self.fc1.bias.data/5
        #self.fc2.bias.data = self.fc2.bias.data/5
        #self.fc3.bias.data = self.fc3.bias.data/5
        #self.fc1.weight.data = self.fc1.weight.data/5
        #self.fc2.weight.data = self.fc2.weight.data/5
        #self.fc3.weight.data = self.fc3.weight.data/5

    def forward(self, inp):
        #return self.sgmd(self.fc2(self.sgmd(self.fc1(inp))))
        #return self.fc2(self.sgmd(self.fc1(inp)))
        return self.fc3(self.sgmd(self.fc2(self.rlu(self.fc1(inp)))))


    def features_num(self, inp):
        size = inp.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features = num_features * s
        return num_features


class LogisticR(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticR, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    def forward(self, x):
        #out = F.sigmoid(self.linear(x))
        out = self.linear(x)
        return out
