from SCN import SCN
from Fractal_generator import koch, binary_frac
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pickle
from scipy.stats import multivariate_normal
from Julia_Set import Julia


X = np.arange(0.00, 0.50, 0.01)
Y = np.arange(0.00, 0.50, 0.01)
X, Y = np.meshgrid(X, Y)


pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
# The distribution on the variables X, Y packed into pos.

Z = Julia()


visible_units = Variable(torch.FloatTensor([[0,0], [1,0], [0,1]]).view(3, -1))



batch_size = 1000
input_dim = 1

iterations = 10000
experiments = 1
lr1 = 0.01

S = np.zeros(iterations)

for experiment in range(experiments):
    scn = SCN(3, 2, visible_units, 4)
    optimizer = torch.optim.SGD(scn.parameters(), lr=lr1)
    criterion = torch.nn.MSELoss()
    pos1 = torch.from_numpy(pos)
    pos1 = pos1.type(torch.FloatTensor)
    Z1 = torch.from_numpy(Z)
    Z1 = Z1.type(torch.FloatTensor)
    for i in range(iterations):
        sample_inds = np.random.choice(pos1.size()[0], batch_size)
        sample_inds2 = np.random.choice(pos1.size()[1], batch_size)
        samples = Variable(pos1[sample_inds, sample_inds2])
        y = Variable(Z1[sample_inds, sample_inds2])
        output = scn(samples).view(-1, 1)
        loss = criterion(output, y)
        S[i] += loss.data[0]
        loss.backward(retain_graph=True)
        optimizer.step()
        volatility = 1
        for j in range(scn.depth):
            scn.L[j].data = (scn.L[j].data - lr1*volatility * scn.L[j].grad.data).clamp(0.4,0.6)
            scn.L[j].data = (scn.L[j].data / (scn.L[j].data.sum())).clamp(0, 1)
            #scn.L[j].data = torch.ones(scn.L[j].size()) / 2
            volatility*= 1
        #scn.visible_fs.data = torch.zeros(scn.visible_fs.size())
        #scn.biases.data = torch.zeros(scn.biases.size())
        optimizer.zero_grad()

        if i % 300 == 0:
            print i
            plt.close()
            #pltx = pos1.view(-1, input_dim).numpy()
            plty1 = scn(Variable(pos1.view(-1, 2))).data.view(50,50)
            #plty = Z1.view(-1, 1).numpy()
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            #surf = ax.plot_surface(X, Y, plty1.numpy(), cmap=cm.viridis, linewidth=0, antialiased=False)
            #surf2 = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            cset = ax.contourf(X, Y, plty1.numpy(), zdir='z', offset=0.05, cmap=cm.viridis)
            plt.pause(0.5)
            plt.cla()

with open("scn_res.txt2", "wb") as fp:  # Pickling
    pickle.dump(S/experiments, fp)

#plt.plot(range(iterations), S)
plt.show()
