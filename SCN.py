#### SIMPLICIAL COMPLEX NEURAL NETWORK LEARN FRACTALS

import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class SCN(torch.nn.Module):

    def __init__(self, visible_num, input_dim, visible_units, depth, model=1):
        super(SCN, self).__init__()
        # depth = number of hidden units
        self.L = []
        for _ in range(depth):
            self.L.append(torch.nn.Parameter(torch.ones(1,visible_num)/visible_num, requires_grad=True))
            ## visible units are defined as columns of a matrix
        ### UNCOMMENT THESE TWO LINES FOR SCN_fractal_test
        #self.visible_fs = torch.nn.Parameter(torch.randn(visible_num, 1)/5, requires_grad=True)
        #self.biases = torch.nn.Parameter(torch.randn(depth, 1)/5, requires_grad=True)
        ###
        self.visible_fs = torch.nn.Parameter(torch.zeros(visible_num, 1), requires_grad = True)
        self.biases = torch.nn.Parameter(torch.zeros(depth,1), requires_grad = True)
        self.visible_units = visible_units

        self.depth = depth
        self.visible_num = visible_num
        self.input_dim = input_dim
        ## for one dimensional data
        self.model = model


    def forward(self, inp):
        hidden_collect = []
        f = self.visible_fs.repeat(inp.size()[0], 1, 1)
        h = self.visible_units.repeat(inp.size()[0], 1, 1)
        input_weights = self.get_first_input_weights_mdl1(inp, self.visible_units)
        for i in range(self.depth):
            input_weights, indices = self.update_weights(input_weights, self.L[i])
            h_old = h.clone()
            if self.model == 1:
                f, h = self.update_h_mdl1(f, h, indices, self.L[i], i)
            elif self.model == 2:
                f, h = self.update_h_mdl2(f, h, indices, self.L[i], i)
            hidden_collect.append([h_old, h[range(h.size()[0]), indices.long(), :]])
        out = torch.bmm(input_weights.view(inp.size()[0], 1, -1), f)
        #out = torch.nn.Sigmoid()(torch.bmm(input_weights.view(inp.size()[0], 1, -1), f))
        return out, hidden_collect


    def update_weights(self, inp_weights, h_w):
        weights_div = inp_weights / (h_w + 1e-20)
        values, indices = torch.min(weights_div, 1)
        input_weights = self.update_inp_weights_mdl1(values, indices, inp_weights, h_w)
        return input_weights, indices

    def update_inp_weights_mdl1(self, values, indices, inp_weights, h_w):
        new_weights = inp_weights - values.view(-1,1).repeat(1,self.visible_num) * h_w
        new_weights[range(values.size()[0]),indices.long()] = values
        return new_weights

    def update_h_mdl1(self, f, h, indices, weights, i):
        new_h = torch.bmm(weights.repeat(h.size()[0], 1, 1), h).view(-1, self.input_dim)
        f[range(h.size()[0]), indices.long(), :] = (torch.bmm(weights.repeat(h.size()[0], 1, 1), f.clone())
                                                    + self.biases[i]).squeeze(-1)
        h[range(h.size()[0]), indices.long(), :] = new_h
        return f, h

    def update_h_mdl2(self, f, h, indices, weights, i):
        new_h = torch.bmm(weights.repeat(h.size()[0], 1, 1), h).view(-1, self.input_dim)
        f[range(h.size()[0]), indices.long(), :] = \
            torch.bmm(weights.repeat(h.size()[0], 1, 1), f.clone()) + torch.nn.Sigmoid()(self.biases[i])
        h[range(h.size()[0]), indices.long(), :] = new_h
        return f, h

    def get_first_input_weights_mdl1(self, inp, visible_units):
        return torch.cat((1 - torch.sum(inp, 1).view(-1, 1), inp), 1)

    def features_num(self, inp):
        size = inp.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features = num_features * s
        return num_features

    def initial_weights(self):
        return 1


    def project_simplex(self, v, z=1): #sparsemax
        v_sorted, _ = torch.sort(v, dim=0, descending=True)
        cssv = torch.cumsum(v_sorted, dim=0) - z
        ind = torch.arange(1, 1 + len(v)).to(dtype=v.dtype)
        cond = v_sorted - cssv / ind > 0
        rho = ind.masked_select(cond)[-1]
        tau = cssv.masked_select(cond)[-1] / rho
        w = torch.clamp(v - tau, min=0)
        return w


batch_size = 10
input_dim = 1

iterations = 10000
lr1 = 0.01

if __name__ == "__main__":
    X = torch.from_numpy(np.arange(100)*0.01 + 0.01).view(100, -1)
    X = X.type(torch.FloatTensor)
    #Y = ((X - 0.5) * (X - 0.5) + torch.rand(100,1)/50).view(100, -1)
    Y = torch.from_numpy(norm.pdf(np.arange(100)*0.01 + 0.01,loc=0.7, scale=0.07)).view(100,-1) + \
        torch.from_numpy(norm.pdf(np.arange(100)*0.01 + 0.01, loc=0.3, scale=0.07)).view(100, -1)
    Y = Y.type(torch.FloatTensor)

    visible_units = Variable(torch.FloatTensor([0, 1]).view(2, -1))
    #visible_init_y = Variable(torch.FloatTensor([0, 0]).view(-1, 2))

    scn = SCN(2, 1, visible_units, 30)



    optimizer = torch.optim.SGD(scn.parameters(), lr=lr1)
    criterion = torch.nn.MSELoss()
    for i in range(iterations):
        sample_inds = np.random.choice(X.size()[0], batch_size)
        samples = Variable(X[sample_inds])
        y = Variable(Y[sample_inds])
        output = scn(samples).view(-1,1)
        loss = criterion(output, y)
        loss.backward(retain_graph = True)

        optimizer.step()
        volatility = 1
        for j in range(scn.depth):
            #scn.L[j].data = (scn.L[j].data - lr1*volatility*scn.L[j].grad.data).clamp(0.4,0.6)
            #scn.L[j].data = (scn.L[j].data/(scn.L[j].data.sum())).clamp(0,1)
            scn.L[j].data = torch.ones(scn.L[j].size())/2
            #volatility *= 0.9
        optimizer.zero_grad()

        #scn.biases.data = torch.zeros(scn.biases.size())
        #scn.visible_fs.data = torch.zeros(scn.visible_fs.size())



    #print(scn(Variable(torch.FloatTensor([0.5]).view(1,1))))
    #print(scn.L[0])
        if i% 10 == 0:
            pltx = X.view(-1,input_dim).numpy()
            plty1 = scn(Variable(X)).data.view(-1,1).numpy()
            plty = Y.view(-1,1).numpy()

            plt.plot(pltx, plty, pltx, plty1)
            plt.xlim(0,1.01)
            plt.pause(0.2)
            plt.clf()

plt.show()



