import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

class FC(torch.nn.Module):

    def __init__(self, input_dim):
        super(FC, self).__init__()
        # depth = number of hidden units
        self.fc1 = torch.nn.Linear(input_dim, 300)
        self.rlu = torch.nn.ReLU()
        self.sgmd = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(300, 1)
        self.fc3 = torch.nn.Linear(300,1)
        #self.fc1.data = torch.zeros(input_dim, 300)
        #self.fc2.data = torch.zeros(300, 1)
        #self.fc3.data = torch.zeros(300,1)
        ## for one dimensional data


    def forward(self, inp):
        return self.sgmd(self.fc2(self.rlu(self.fc1(inp))))


    def features_num(self, inp):
        size = inp.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features = num_features * s
        return num_features