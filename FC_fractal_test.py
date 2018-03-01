from SCN import SCN
from Fractal_generator import koch, binary_frac
from Fullyconnected_nets import FC
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pickle



direction = [0.0,float(1)/243]

X, Y = koch([[0,0]], 5, direction)
#X, Y = binary_frac([], 6, 0, 1)
X = torch.from_numpy(np.asarray(X, dtype=np.float32)).view(len(X), -1)
X = X.type(torch.FloatTensor)# + torch.rand(X.size())*1/97
Y = torch.from_numpy(np.asarray(Y, dtype=np.float32)).view(len(Y), -1)



batch_size = 1000
input_dim = 1


experiments = 10
iterations = 5000
lr1 = 0.1



S = np.zeros(iterations)
for experiment in range(experiments):
    fc = FC(1)
    optimizer = torch.optim.SGD(fc.parameters(), lr=lr1)
    criterion = torch.nn.L1Loss()
    for i in range(iterations):
        sample_inds = np.random.choice(X.size()[0], batch_size)
        samples = Variable(X[sample_inds])
        y = Variable(Y[sample_inds])
        output = fc(samples).view(-1, 1)
        loss = criterion(output, y)
        S[i] += loss.data[0]
        loss.backward()
        print loss
        optimizer.step()
        optimizer.zero_grad()
        if i % 100 == 0:
            print i
            pltx = X.view(-1, input_dim).numpy()
            plty1 = fc(Variable(X)).data.view(-1, 1).numpy()
            plty = Y.view(-1, 1).numpy()

            plt.scatter(pltx, plty)
            plt.scatter(pltx, plty1)
            plt.xlim(0, 1)
            plt.pause(0.1)
            plt.clf()


with open("fc_res2.txt", "wb") as fp:  # Pickling
    pickle.dump(S/experiments, fp)

#plt.plot(range(iterations), S)
plt.show()
