import turtle

import numpy as np
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt



def rotation_matrix(angle):
    return [[math.cos(math.pi*angle/180), math.sin(math.pi*angle/180)],
            [-math.sin(math.pi*angle/180), math.cos(math.pi*angle/180)]]



def koch(vec, order, direction):
    if order == 0:
        vec.append([vec[-1][0] + direction[0], vec[-1][1] + direction[1]])
    else:
        koch(vec, order-1, direction)
        direction = np.matmul(rotation_matrix(60), direction)
        koch(vec, order-1, direction)
        direction = np.matmul(rotation_matrix(-120), direction)
        koch(vec, order-1, direction)
        direction = np.matmul(rotation_matrix(60), direction)
        koch(vec, order-1, direction)

    return [row[1] for row in vec], [row[0] for row in vec]



def binary_frac(vec, order, num1, num2):
    if order == 0:
        return
    else:
        mid = (float(num1) + num2)/2
        vec.append([float(2**order)/100, mid])
        binary_frac(vec, order-1, num1, mid)
        binary_frac(vec, order-1, mid, num2)

    return [row[1] for row in vec], [row[0] for row in vec]


#direction = [0.0,1]

#vec =koch([[0,0]], 3, direction)
#X, Y =binary_frac([], 3, 0, 1)
#print(X, Y)
#print(max(vec))
#plt.scatter([row[1] for row in vec], [row[0] for row in vec])
#plt.xlim(0,25)
#plt.ylim(-12,12)
#plt.show()