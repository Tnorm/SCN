from Fractal_generator import koch, binary_frac
import matplotlib.pyplot as plt
import torch
import numpy as np


direction = [0.0,float(1)/243]

#X, Y = koch([[0,0]], 5, direction)
X, Y = binary_frac([], 4, 0, 1)

X2 = [X[idx] for idx in np.argsort(X,-1)]
Y2 = [Y[idx] for idx in np.argsort(X,-1)]

X = X2
Y = Y2

X = torch.from_numpy(np.asarray(X, dtype=np.float32)).view(len(X), -1)
X = X.type(torch.FloatTensor)# + torch.rand(X.size())*1/97
Y = torch.from_numpy(np.asarray(Y, dtype=np.float32)).view(len(Y), -1)


pltx = X.view(-1, 1).numpy()
plty = Y.view(-1, 1).numpy()



for i in range(len(X)-1):
    plt.plot([X[i],X[i+1]], [6.25*Y[i],6.25*Y[i+1]],marker='.', color='chocolate', linewidth=3)

#plt.scatter(pltx, 6.25*plty, color='chocolate', s=5)

#plt.show()
plt.savefig('fig/binary_frac.png')