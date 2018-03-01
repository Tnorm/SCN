from SCN import SCN
from Fractal_generator import koch, binary_frac
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pickle




direction = [0.0,float(1)/243]

X, Y = koch([[0,0]], 5, direction)
#X, Y = binary_frac([], 10, 0, 1)
X = torch.from_numpy(np.asarray(X, dtype=np.float32)).view(len(X), -1)
X = X.type(torch.FloatTensor)# + torch.rand(X.size())*1/97
Y = torch.from_numpy(np.asarray(Y, dtype=np.float32)).view(len(Y), -1)

#print(X, Y)
visible_units = Variable(torch.FloatTensor([0, 1]).view(2, -1))



batch_size = 1000
input_dim = 1

iterations = 5000
experiments = 10
lr1 = 0.1

S = np.zeros(iterations)

for experiment in range(experiments):
    scn = SCN(2, 1, visible_units, 4)
    optimizer = torch.optim.SGD(scn.parameters(), lr=lr1)
    criterion = torch.nn.MSELoss()
    for i in range(iterations):
        sample_inds = np.random.choice(X.size()[0], batch_size)
        samples = Variable(X[sample_inds])
        y = Variable(Y[sample_inds])
        output = scn(samples).view(-1, 1)
        loss = criterion(output, y)
        S[i] += loss.data[0]
        loss.backward(retain_graph=True)
        optimizer.step()
        volatility = 1
        for j in range(scn.depth):
            scn.L[j].data = (scn.L[j].data - lr1*volatility * scn.L[j].grad.data)
            scn.L[j].data = (scn.L[j].data / (scn.L[j].data.sum())).clamp(0, 1)
            volatility*= 1
            #scn.L[j].data = torch.ones(scn.L[j].size()) / 2
        #scn.visible_fs.data = torch.zeros(scn.visible_fs.size())
        #scn.biases.data = torch.zeros(scn.biases.size())
        optimizer.zero_grad()

        if i % 100 == 0:
            print i
            # pltx = X.view(-1, input_dim).numpy()
            # plty1 = scn(Variable(X)).data.view(-1, 1).numpy()
            # plty = Y.view(-1, 1).numpy()
            #
            # plt.scatter(pltx, plty)
            # plt.scatter(pltx, plty1)
            # plt.xlim(0, 1)
            # plt.pause(0.1)
            # plt.clf()

with open("scn_res.txt2", "wb") as fp:  # Pickling
    pickle.dump(S/experiments, fp)

#plt.plot(range(iterations), S)
plt.show()
