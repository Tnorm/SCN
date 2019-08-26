## DO NOT USE FOR PLOTTING! (MESSY FILE)


from Fractal_generator import koch, binary_frac
import matplotlib.pyplot as plt
import torch
import numpy as np
import pickle
from matplotlib.lines import Line2D
from scipy.stats import norm


direction = [0.0,float(1)/243]

#X_, Y_ = koch([[0,0]], 5, direction)
X_, Y_ = binary_frac([], 4, 0, 1)
#X = torch.from_numpy(np.arange(0.0, 1.0, 0.005, dtype=np.float32)).view(len(np.arange(0.0, 1.0, 0.005)), -1)
#Y = torch.from_numpy(np.asarray(norm.pdf(X, 0.05, 0.1)/3 + norm.pdf(X, 0.95, 0.1)/3 + norm.pdf(X, 0.5, 0.2)/3 + norm.pdf(X, 0.35, 0.2)/3 + norm.pdf(X, 0.65, 0.2)/3 - 
#	norm.pdf(X, 0.25, 0.01)/140 - norm.pdf(X, 0.75, 0.01)/140 - norm.pdf(X, 0.5, 0.02)/50 - norm.pdf(X, 1.0, 0.01)/200 - norm.pdf(X, 0.0, 0.01)/200
#	, dtype=np.float32) \
#	).view(len(np.arange(0.0, 1.0, 0.005)),-1)

plty1 = pickle.load(open('scn_resf_3.txt', "rb" ))
pltyfc = pickle.load(open('fc_resf_3.txt', "rb" ))

X2 = [X_[idx] for idx in np.argsort(X_,-1)]
Y2 = [Y_[idx] for idx in np.argsort(X_,-1)]
#plty1 = [plty1[idx] for idx in np.argsort(X_,-1)]
#pltyfc = [pltyfc[idx]*6.25 for idx in np.argsort(X_,-1)]


#X = X2
#Y = Y2

#X = torch.from_numpy(np.asarray(X, dtype=np.float32)).view(len(X), -1)
X = X.type(torch.FloatTensor)# + torch.rand(X.size())*1/97
#Y = torch.from_numpy(np.asarray(Y, dtype=np.float32)).view(len(Y), -1)


pltx = X.view(-1, 1).numpy()
plty = Y.view(-1, 1).numpy()


#plt.scatter(X.view(-1, 1).numpy(), 6.25*Y.view(-1, 1).numpy(), color='black', s=0.9)
plt.plot(pltx, plty,marker='.', color='black', linewidth=2.7, alpha=1.0, label='s1', markersize=0.5)
plt.plot(pltx, plty1,marker='.', color='orange', linewidth=2.7, alpha=1.0, label='s2', markersize=0.5)
#for i in range(len(X)-1):			
    #plt.plot([X[i],X[i+1]], [6.25*Y[i],6.25*Y[i+1]],marker='.', color='black', linewidth=1.0, alpha=1.0, label='s1')
    #plt.plot([X[i],X[i+1]], [6.25*plty1[i],6.25*plty1[i+1]],marker='.', color='orange', linewidth=2.0, alpha=1.0, label='s2')
    
plt.plot(pltx, pltyfc, color='blue', label='s3', alpha=1.0, linewidth=2.7)

custom_lines = [Line2D([0], [0], color='black', lw=2),
                Line2D([0], [0], color='blue', lw=2),
                Line2D([0], [0], color='orange', lw=2)]
                
plt.legend(custom_lines, ['Target','Fully Connected Net', 'SCN'], prop={'size': 11})


#plt.scatter(pltx, 6.25*plty, color='chocolate', s=5)

#plt.show()
plt.savefig('fig/crazy_func.png')
