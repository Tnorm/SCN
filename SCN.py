#### SIMPLICIAL COMPLEX NEURAL NETWORK LEARN FUNCTIONS!

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
        #out = torch.bmm(input_weights.view(inp.size()[0], 1, -1), f)
        out = torch.nn.Sigmoid()(torch.bmm(input_weights.view(inp.size()[0], 1, -1), f))
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


class SCN_multi(torch.nn.Module):

    def __init__(self, visible_num, input_dim, output_dim, visible_units, depth, model=1):
        super(SCN_multi, self).__init__()
        # depth = number of hidden units
        self.L = torch.nn.ParameterList()
        for _ in range(depth):
            #self.L.append(torch.nn.Parameter(torch.ones(1,visible_num)/visible_num, requires_grad=True))
            self.L.append(torch.nn.Parameter(torch.zeros(1, visible_num), requires_grad=True))
            ## visible units are defined as columns of a matrix
        ### UNCOMMENT THESE TWO LINES FOR SCN_fractal_test
        #self.visible_fs = torch.nn.Parameter(torch.randn(visible_num, 1)/5, requires_grad=True)
        #self.biases = torch.nn.Parameter(torch.randn(depth, 1)/5, requires_grad=True)
        ###
        self.visible_fs = torch.nn.Parameter(torch.zeros(visible_num, output_dim), requires_grad = True)
        self.biases = torch.nn.Parameter(torch.zeros(depth,output_dim), requires_grad = True)
        self.bias_funcs = torch.nn.ModuleList([])
        for _ in range(depth):
            self.bias_funcs.append(torch.nn.Linear(input_dim, output_dim))
            self.bias_funcs[-1].weight.data.fill_(0.0)
        self.visible_units = visible_units

        self.depth = depth
        self.visible_num = visible_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        ## for one dimensional data
        self.model = model

        self.sftmax = torch.nn.Softmax(dim=-1)

    def forward(self, inp):

        hidden_collect = []
        f = self.visible_fs.repeat(inp.size()[0], 1, 1)
        h = self.visible_units.repeat(inp.size()[0], 1, 1)
        input_weights = self.get_first_input_weights_mdl1(inp, self.visible_units)
        for i in range(self.depth):
            input_weights, indices = self.update_weights(input_weights, self.sftmax(self.L[i]).detach())
            h_old = h.clone()
            if self.model == 1:
                f, h, last_h = self.update_h_mdl1(f, h, indices, self.sftmax(self.L[i]), i)
            elif self.model == 2:
                f, h = self.update_h_mdl2(f, h, indices, self.sftmax(self.L[i]), i)
            hidden_collect.append([h_old, h[range(h.size()[0]), indices.long(), :]])
        out = torch.bmm(input_weights.view(inp.size()[0], 1, -1), f)
        #out = torch.nn.Softmax(dim=-1)(torch.bmm(input_weights.view(inp.size()[0], 1, -1), f))
        return out, hidden_collect, last_h

    def update_weights(self, inp_weights, h_w):
        weights_div = inp_weights / (h_w + 1e-20)
        values, indices = torch.min(weights_div, 1)
        input_weights = self.update_inp_weights_mdl1(values, indices, inp_weights, h_w)
        return input_weights, indices

    def update_inp_weights_mdl1(self, values, indices, inp_weights, h_w):
        #new_weights = inp_weights - values.view(-1,1).repeat(1,self.visible_num) * h_w
        new_weights = inp_weights - values.view(-1, 1) * h_w
        new_weights[range(values.size()[0]),indices.long()] = values
        return new_weights

    def update_h_mdl0(self, f, h, indices, weights, i):
        new_h = torch.bmm(weights.repeat(h.size()[0], 1, 1), h).view(-1, self.input_dim)
        f[range(h.size()[0]), indices.long(), :] = (torch.bmm(weights.repeat(h.size()[0], 1, 1), f.clone()))\
                                                       .squeeze(-2) + self.biases[i]
        h[range(h.size()[0]), indices.long(), :] = new_h
        return f, h

    def update_h_mdl1(self, f, h, indices, weights, i):
        new_h = torch.matmul(weights, h.clone()).squeeze(-2)
            #print(new_h[20])
        # if i == 1:
        #     print(new_h, indices)
        # plt.imshow((new_h[5]).reshape(28, 28).data.numpy(), cmap='gray', vmin=0.0,
        #            vmax=new_h[5].max())
        # plt.savefig('hidden' + str(i) + '.png')
        # plt.clf()
        #print(new_h[0])
        f[range(h.size()[0]), indices.long(), :] = torch.matmul(weights, f.clone()).squeeze(-2) + \
                                                   self.biases[i]
        s = torch.autograd.grad(f.sum(), weights)
        print(s)
                                                   #self.bias_funcs[i](new_h.detach())
        h[range(h.size()[0]), indices.long(), :] = new_h
        return f, h, new_h

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



# #### SIMPLICIAL COMPLEX NEURAL NETWORK LEARN FUNCTIONS!
#
# import torch
# from torch.autograd import Variable
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm
#
# class SCN(torch.nn.Module):
#
#     def __init__(self, visible_num, input_dim, visible_units, depth, model=1):
#         super(SCN, self).__init__()
#         # depth = number of hidden units
#         self.L = []
#         for _ in range(depth):
#             self.L.append(torch.nn.Parameter(torch.ones(1,visible_num)/visible_num, requires_grad=True))
#             ## visible units are defined as columns of a matrix
#         ### UNCOMMENT THESE TWO LINES FOR SCN_fractal_test
#         #self.visible_fs = torch.nn.Parameter(torch.randn(visible_num, 1)/5, requires_grad=True)
#         #self.biases = torch.nn.Parameter(torch.randn(depth, 1)/5, requires_grad=True)
#         ###
#         self.visible_fs = torch.nn.Parameter(torch.zeros(visible_num, 1), requires_grad = True)
#         self.biases = torch.nn.Parameter(torch.zeros(depth,1), requires_grad = True)
#         self.visible_units = visible_units
#
#         self.depth = depth
#         self.visible_num = visible_num
#         self.input_dim = input_dim
#         ## for one dimensional data
#         self.model = model
#
#
#     def forward(self, inp):
#         hidden_collect = []
#         f = self.visible_fs.repeat(inp.size()[0], 1, 1)
#         h = self.visible_units.repeat(inp.size()[0], 1, 1)
#         input_weights = self.get_first_input_weights_mdl1(inp, self.visible_units)
#         for i in range(self.depth):
#             input_weights, indices = self.update_weights(input_weights, self.L[i])
#             h_old = h.clone()
#             if self.model == 1:
#                 f, h = self.update_h_mdl1(f, h, indices, self.L[i], i)
#             elif self.model == 2:
#                 f, h = self.update_h_mdl2(f, h, indices, self.L[i], i)
#             hidden_collect.append([h_old, h[range(h.size()[0]), indices.long(), :]])
#         #out = torch.bmm(input_weights.view(inp.size()[0], 1, -1), f)
#         out = torch.nn.Sigmoid()(torch.bmm(input_weights.view(inp.size()[0], 1, -1), f))
#         return out, hidden_collect
#
#
#     def update_weights(self, inp_weights, h_w):
#         weights_div = inp_weights / (h_w + 1e-20)
#         values, indices = torch.min(weights_div, 1)
#         input_weights = self.update_inp_weights_mdl1(values, indices, inp_weights, h_w)
#         return input_weights, indices
#
#     def update_inp_weights_mdl1(self, values, indices, inp_weights, h_w):
#         new_weights = inp_weights - values.view(-1,1).repeat(1,self.visible_num) * h_w
#         new_weights[range(values.size()[0]),indices.long()] = values
#         return new_weights
#
#     def update_h_mdl1(self, f, h, indices, weights, i):
#         new_h = torch.bmm(weights.repeat(h.size()[0], 1, 1), h).view(-1, self.input_dim)
#         f[range(h.size()[0]), indices.long(), :] = (torch.bmm(weights.repeat(h.size()[0], 1, 1), f.clone())
#                                                     + self.biases[i]).squeeze(-1)
#         h[range(h.size()[0]), indices.long(), :] = new_h
#         return f, h
#
#     def update_h_mdl2(self, f, h, indices, weights, i):
#         new_h = torch.bmm(weights.repeat(h.size()[0], 1, 1), h).view(-1, self.input_dim)
#         f[range(h.size()[0]), indices.long(), :] = \
#             torch.bmm(weights.repeat(h.size()[0], 1, 1), f.clone()) + torch.nn.Sigmoid()(self.biases[i])
#         h[range(h.size()[0]), indices.long(), :] = new_h
#         return f, h
#
#     def get_first_input_weights_mdl1(self, inp, visible_units):
#         return torch.cat((1 - torch.sum(inp, 1).view(-1, 1), inp), 1)
#
#     def features_num(self, inp):
#         size = inp.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features = num_features * s
#         return num_features
#
#     def initial_weights(self):
#         return 1
#
#
#     def project_simplex(self, v, z=1): #sparsemax
#         v_sorted, _ = torch.sort(v, dim=0, descending=True)
#         cssv = torch.cumsum(v_sorted, dim=0) - z
#         ind = torch.arange(1, 1 + len(v)).to(dtype=v.dtype)
#         cond = v_sorted - cssv / ind > 0
#         rho = ind.masked_select(cond)[-1]
#         tau = cssv.masked_select(cond)[-1] / rho
#         w = torch.clamp(v - tau, min=0)
#         return w
#
#
# class SCN_multi(torch.nn.Module):
#
#     def __init__(self, visible_num, input_dim, output_dim, visible_units, depth, model=1):
#         super(SCN_multi, self).__init__()
#         # depth = number of hidden units
#         self.L = torch.nn.ParameterList()
#         for _ in range(depth):
#             self.L.append(torch.nn.Parameter(torch.ones(1,visible_num)/visible_num, requires_grad=True))
#             ## visible units are defined as columns of a matrix
#         ### UNCOMMENT THESE TWO LINES FOR SCN_fractal_test
#         #self.visible_fs = torch.nn.Parameter(torch.randn(visible_num, 1)/5, requires_grad=True)
#         #self.biases = torch.nn.Parameter(torch.randn(depth, 1)/5, requires_grad=True)
#         ###
#         self.visible_fs = torch.nn.Parameter(torch.zeros(visible_num, output_dim), requires_grad = True)
#         self.biases = torch.nn.Parameter(torch.zeros(depth,output_dim), requires_grad = True)
#         self.bias_funcs = torch.nn.ModuleList([])
#         for _ in range(depth):
#             self.bias_funcs.append(torch.nn.Linear(input_dim, output_dim))
#             self.bias_funcs[-1].weight.data.fill_(0.0)
#         self.visible_units = visible_units
#
#         self.depth = depth
#         self.visible_num = visible_num
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         ## for one dimensional data
#         self.model = model
#
#
#     def forward(self, inp):
#         hidden_collect = []
#         f = self.visible_fs.repeat(inp.size()[0], 1, 1)
#         h = self.visible_units.repeat(inp.size()[0], 1, 1)
#         input_weights = self.get_first_input_weights_mdl1(inp, self.visible_units)
#         for i in range(self.depth):
#             input_weights, indices = self.update_weights(input_weights, self.L[i])
#             h_old = h.clone()
#             if self.model == 1:
#                 f, h = self.update_h_mdl1(f, h, indices, self.L[i], i)
#             elif self.model == 2:
#                 f, h = self.update_h_mdl2(f, h, indices, self.L[i], i)
#             hidden_collect.append([h_old, h[range(h.size()[0]), indices.long(), :]])
#         out = torch.bmm(input_weights.view(inp.size()[0], 1, -1), f)
#         #out = torch.nn.Softmax(dim=-1)(torch.bmm(input_weights.view(inp.size()[0], 1, -1), f))
#         return out, hidden_collect
#
#     def update_weights(self, inp_weights, h_w):
#         weights_div = inp_weights / (h_w + 1e-20)
#         values, indices = torch.min(weights_div, 1)
#         #print(indices)
#         input_weights = self.update_inp_weights_mdl1(values, indices, inp_weights, h_w)
#         return input_weights, indices
#
#     def update_inp_weights_mdl1(self, values, indices, inp_weights, h_w):
#         #new_weights = inp_weights - values.view(-1,1).repeat(1,self.visible_num) * h_w
#         new_weights = inp_weights - values.view(-1, 1) * h_w
#         new_weights[range(values.size()[0]),indices.long()] = values
#         return new_weights
#
#     def update_h_mdl0(self, f, h, indices, weights, i):
#         new_h = torch.bmm(weights.repeat(h.size()[0], 1, 1), h).view(-1, self.input_dim)
#         f[range(h.size()[0]), indices.long(), :] = (torch.bmm(weights.repeat(h.size()[0], 1, 1), f.clone()))\
#                                                        .squeeze(-2) + self.biases[i]
#         h[range(h.size()[0]), indices.long(), :] = new_h
#         return f, h
#
#     def update_h_mdl1(self, f, h, indices, weights, i):
#         new_h = torch.matmul(weights, h).squeeze()
#             #print(new_h[20])
#         # if i == 1:
#         #     print(new_h, indices)
#         plt.imshow((new_h[3]).reshape(28, 28).data.numpy(), cmap='gray', vmin=0.0,
#                    vmax=new_h[3].max())
#         plt.savefig('hidden' + str(i) + '.png')
#         plt.clf()
#         f[range(h.size()[0]), indices.long(), :] = torch.matmul(weights.detach(), f.clone()).squeeze() + \
#                                                    self.bias_funcs[i](new_h.detach())
#         h[range(h.size()[0]), indices.long(), :] = new_h
#         return f, h
#
#     def update_h_mdl2(self, f, h, indices, weights, i):
#         new_h = torch.bmm(weights.repeat(h.size()[0], 1, 1), h).view(-1, self.input_dim)
#         f[range(h.size()[0]), indices.long(), :] = \
#             torch.bmm(weights.repeat(h.size()[0], 1, 1), f.clone()) + torch.nn.Sigmoid()(self.biases[i])
#         h[range(h.size()[0]), indices.long(), :] = new_h
#         return f, h
#
#     def get_first_input_weights_mdl1(self, inp, visible_units):
#         return torch.cat((1 - torch.sum(inp, 1).view(-1, 1), inp), 1)
#
#     def features_num(self, inp):
#         size = inp.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features = num_features * s
#         return num_features
#
#     def initial_weights(self):
#         return 1
#
#
#     def project_simplex(self, v, z=1): #sparsemax
#         v_sorted, _ = torch.sort(v, dim=0, descending=True)
#         cssv = torch.cumsum(v_sorted, dim=0) - z
#         ind = torch.arange(1, 1 + len(v)).to(dtype=v.dtype)
#         cond = v_sorted - cssv / ind > 0
#         rho = ind.masked_select(cond)[-1]
#         tau = cssv.masked_select(cond)[-1] / rho
#         w = torch.clamp(v - tau, min=0)
#         return w
