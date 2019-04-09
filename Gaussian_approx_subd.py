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
import matplotlib.lines as mlines

from matplotlib.pyplot import figure

#figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')


def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l


X = np.arange(0.01, 0.5, 0.01)
Y = np.arange(0.01, 0.5, 0.01)
X, Y = np.meshgrid(X, Y)


mu = np.array([0.4, 0.4])
sigma = np.array([[1., 0], [0., 1.]])/400

mu_2 = np.array([0.4, 0.1])
sigma_2 = np.array([[1., 0], [0., 1.]])/400

mu_3 = np.array([0.1, 0.4])
sigma_3 = np.array([[1., 0], [0., 1.]])/400

mu_4 = np.array([0.1, 0.1])
sigma_4 = np.array([[1., 0], [0., 1.]])/400

pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

F = multivariate_normal(mu, sigma)

F_2 = multivariate_normal(mu_2, sigma_2)
F_3 = multivariate_normal(mu_3, sigma_3)
F_4 = multivariate_normal(mu_4, sigma_4)
# The distribution on the variables X, Y packed into pos.
Z = F.pdf(pos) + F_2.pdf(pos) + F_3.pdf(pos)# + F_4.pdf(pos)


visible_units = Variable(torch.FloatTensor([[0,0], [1,0], [0,1]]).view(3, -1))



batch_size = 1000
input_dim = 1

iterations = 200000
experiments = 1
lr1 = 0.1

S = np.zeros(iterations)

#plt.plot([0,0],[0,1] ,marker='o', color='brown')
#plt.plot([0,1],[0,0] ,marker='o', color='brown')
#plt.plot([0,1],[1,0] ,marker='o', color='brown')

depth = 10
color_list = ['black', 'black', 'black', 'black', 'black']
decay = torch.from_numpy(np.exp(-np.arange(0, depth, dtype=np.float))).float()
loss = float('inf')
for experiment in range(experiments):
    scn = SCN(3, 2, visible_units, depth)
    optimizer = torch.optim.SGD(scn.parameters(), lr=lr1)
    criterion = torch.nn.MSELoss()
    pos1 = torch.from_numpy(pos)
    pos1 = pos1.type(torch.FloatTensor)
    Z1 = torch.from_numpy(Z)
    Z1 = Z1.type(torch.FloatTensor)
    for i in range(iterations):
        if i % 500 == 0:
            print(i, loss)
            #pltx = pos1.view(-1, input_dim).numpy()
            plty1, hiddencollect = scn(Variable(pos1.view(-1, 2)))
            plty1 = plty1.data.view(49,49)
            #plty = Z1.view(-1, 1).numpy()
            #print(hiddencollect[0][1])


            # for dep in range(depth):
            #     for sampnum in range(0,2401,8):
            #         for vert in range(3):
            #             plt.plot([hiddencollect[dep][0][sampnum][vert][0].data.numpy(), hiddencollect[dep][1][sampnum][0].data.numpy()],
            #                 [hiddencollect[dep][0][sampnum][vert][1].data.numpy(),hiddencollect[dep][1][sampnum][1].data.numpy()],
            #                  marker='o', color=color_list[dep])


            #newline(hiddencollect[0][0][10][0].data.numpy(), hiddencollect[0][1][10].data.numpy())
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            surf = ax.plot_surface(X, Y, plty1.numpy(), cmap=cm.Oranges, linewidth=0, antialiased=False)
            #surf2 = ax.plot_surface(X, Y, Z, cmap=cm.Oranges, linewidth=0, antialiased=False)
            #cset = ax.contourf(X, Y, plty1.numpy(), zdir='z', offset=0.05, cmap=cm.viridis)
            plt.pause(0.5)
            plt.clf()
            #plt.show()
            plt.close(fig)

        sample_inds = np.random.choice(pos1.size()[0], batch_size)
        sample_inds2 = np.random.choice(pos1.size()[1], batch_size)
        samples = Variable(pos1[sample_inds, sample_inds2])
        y = Variable(Z1[sample_inds, sample_inds2]).view(-1,1)
        output, _ = scn(samples)
        output = output.view(-1, 1)
        loss = criterion(output, y)
        S[i] += loss.data.item()
        loss.backward(retain_graph=True)
        volatility = 1
        for j in range(scn.depth):
            scn.L[j].data = (scn.L[j].data - lr1 * volatility * scn.L[j].grad.data).clamp(0.4, 0.6)
            scn.L[j].data = scn.project_simplex(scn.L[j].data.view(3)).view(1, 3)
            #scn.L[j].data = (scn.L[j].data - lr1*volatility * scn.L[j].grad.data).clamp(0.4,0.6)
            #scn.L[j].data = (scn.L[j].data / (scn.L[j].data.sum())).clamp(0, 1)
            #scn.L[j].data = torch.ones(scn.L[j].size()) / 2
            scn.biases.grad.data[j] = scn.biases.grad.data[j] * decay[j]
            volatility*= 1

        optimizer.step()
        #scn.visible_fs.data = torch.zeros(scn.visible_fs.size())
        #scn.biases.data = torch.zeros(scn.biases.size())
        optimizer.zero_grad()

with open("scn_res2.txt", "wb") as fp:  # Pickling
    pickle.dump(S/experiments, fp)

#plt.plot(range(iterations), S)
#plt.savefig('gaussapp_target.png')
plt.show()
