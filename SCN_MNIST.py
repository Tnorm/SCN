from SCN import SCN
from Fractal_generator import koch, binary_frac
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as dset
import torchvision.transforms as transforms


root = 'data'
download = False

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dset.MNIST(root=root, train=False, transform=trans, download=download)

dataiter = iter(train_set)
images, labels = dataiter.next()

print(torch.max(images[0] + 0.5))
#plt.imshow(images[0], cmap='gray')
#plt.show()


zero = Variable(torch.zeros(1,28*28))
eye = Variable(torch.eye(28*28))
visible_units = torch.cat((zero, eye), 0)

#visible_units = Variable(torch.FloatTensor([0, 1]).view(2, -1))
scn = SCN(28*28+1, 28*28, visible_units, 3, model=2)

batch_size = 1
input_dim = 1

iterations = 10000
lr1 = .0001
optimizer = torch.optim.SGD(scn.parameters(), lr=lr1)
criterion = torch.nn.MSELoss()
for i in range(iterations):
    samples = Variable((images[0] + 0.5)/torch.sum(images[0] + 0.5)).view(1,-1)
    y = Variable(torch.zeros(1,1))
    output = scn(samples)[0].view(-1, 1)
    loss = criterion(output, y)
    loss.backward(retain_graph=True)

    optimizer.step()
    volatility = 1
    for j in range(scn.depth):
        scn.L[j].data = scn.L[j].data - lr1*volatility * scn.L[j].grad.data
        scn.L[j].data = (scn.L[j].data / (scn.L[j].data.sum())).clamp(0, 1)
        volatility*= 0.9
        #scn.L[j].data = torch.ones(scn.L[j].size()) / 2
    if i % 100 == 0:
        print scn(samples)

    optimizer.zero_grad()

#visible_units = Variable(torch.FloatTensor([0, 1]).view(2, -1))
#scn = SCN(2, 1, visible_units, 1)

