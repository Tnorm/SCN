import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

pickle_in = open("scn_res.txt","rb")
scn_loss = pickle.load(pickle_in)

pickle_in = open("fc_res.txt","rb")
fc_loss = pickle.load(pickle_in)

print(scn_loss, fc_loss)

fc_loss = savgol_filter(fc_loss, 3,0)

num = 1000

plt.plot([x*25 for x in range(num)], np.log(fc_loss)[range(0, 25000, 25)])
plt.plot([x*25 for x in range(num)], np.log(scn_loss)[range(0, 25000, 25)])
plt.legend(labels=['Fully Connected Net', 'SCN'], fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Log Loss', fontsize=14)
#plt.show()ss
plt.savefig('comparison_2.png')