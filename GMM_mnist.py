## GMM with mega large number of components. You know what? Poop in mars.

import tensorflow as tf
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
#load the data

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', validation_size=0)

#considering only first 2 data points
img = mnist.train.images[:5000]
print(img.shape)

gmm = GaussianMixture(n_components=1000, covariance_type='diag').fit(img)
samples = gmm.sample(5)

samples = samples[0].reshape((5, 28, 28))

plt.imshow(samples[0])
plt.show()