import pickle
import matplotlib.pyplot as plt
import numpy as np


pickle_in = open("scn_res.txt","rb")
scn_loss = pickle.load(pickle_in)

pickle_in = open("fc_res2.txt","rb")
fc_loss = pickle.load(pickle_in)


num = 3000

plt.plot(range(num), np.log(scn_loss)[0:num],
         range(num), np.log(fc_loss)[0:num])
plt.show()