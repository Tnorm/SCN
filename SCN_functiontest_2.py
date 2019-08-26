from SCN import SCN, SCN_multi
from Fractal_generator import koch, binary_frac
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import norm
import time



X = np.linspace(0.01, 0.99, 99)
X = torch.from_numpy(np.asarray(X, dtype=np.float32)).view(len(X), -1)

X = X.type(torch.FloatTensor)
Y = torch.from_numpy(np.asarray(np.absolute(np.linspace(0.01, 0.99, 99) - 0.2) *
                     ((np.linspace(0.01, 0.99, 99)<0.2) +
                      (np.linspace(0.01, 0.99, 99)>0.2)*0.25), dtype=np.float32)).view(len(X), -1)

visible_units = Variable(torch.FloatTensor([0.0, 1.0]).view(2, -1))

batch_size = 99
input_dim = 1

iterations = 200000
experiments = 10
lr1 = 1e-4

S = np.zeros(X.size()[0])

for experiment in range(experiments):
    scn = SCN_multi(2, 1, 1, visible_units, 1)
    optimizer = torch.optim.Adam(scn.parameters(), lr=lr1)
    criterion = torch.nn.MSELoss()
    scn.visible_fs.data = torch.ones(2, 1) * 0.2
    scn.biases.data = torch.zeros(1,1) - 0.2
    for i in range(iterations):
        sample_inds = np.random.choice(X.size()[0], batch_size)
        samples = Variable(X[sample_inds])
        y = Variable(Y[sample_inds])
        output = scn(samples)[0].view(-1, 1)
        loss = criterion(output, y)
        # S[i] += loss.data[0]
        loss.backward(retain_graph=True)
        scn.visible_fs.grad.data.fill_(0.0)
        scn.biases.grad.data.fill_(0.0)
        print(scn.L[0].grad.data)
        optimizer.step()

        if i % 1000 == 0:
            print(i)
            pltx = X.view(-1, input_dim).numpy()
            plty1 = scn(Variable(X))[0].data.view(-1, 1).numpy()
            plty = Y.view(-1, 1).numpy()
            # print(scn.biases.data)
            plt.scatter(pltx, plty)
            plt.scatter(pltx, plty1)
            # plt.xlim(0, 1)
            plt.pause(0.1)
            plt.close()
        #time.sleep(0.5)
    S = np.add(S, plty1.reshape(S.shape))

with open("scn_resf_3.txt", "wb") as fp:  # Pickling
    pickle.dump(S / experiments, fp)

# plt.plot(range(iterations), S)
plt.show()
