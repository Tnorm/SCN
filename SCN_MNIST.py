from SCN import SCN
from Fractal_generator import koch, binary_frac
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as Data





root = 'data'
download = False


batch_size = 100


trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dset.MNIST(root=root, train=False, transform=trans, download=download)

train_loader = Data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = Data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


#dataiter = iter(train_set)
#images, labels = dataiter.next()


#print(torch.min(images[0] + 0.5))
#plt.imshow(images[0], cmap='gray')
#plt.show()


zero = Variable(torch.zeros(1,28*28))
eye = Variable(torch.eye(28*28))
visible_units = torch.cat((zero, eye), 0)

#visible_units = Variable(torch.FloatTensor([0, 1]).view(2, -1))
scn = SCN(28*28+1, 28*28, visible_units, 10, model=1)

iterations = 50000
epoch = 20
lr1 = .001
optimizer = torch.optim.Adam(scn.parameters(), lr=lr1)
criterion = torch.nn.MSELoss()

S = []
y_onehot = torch.FloatTensor(batch_size, 10)
for ep in range(epoch):
    for i, (samples, y) in enumerate(train_loader):
        samples = Variable(samples).view(-1,784)
        samples = samples + 0.5
        sample_sums = torch.sum(samples, -1)
        samples = samples / sample_sums.view(-1,1).repeat(1,784)
        y = Variable(y).view(-1,1)
        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)
        output = scn(samples)[0].view(-1, 1)
        loss = criterion(output, y_onehot[:,0].view(-1,1))
        loss.backward(retain_graph=True)
        optimizer.step()
        volatility = 1
        for j in range(scn.depth):
            scn.L[j].data = scn.L[j].data - lr1*volatility * scn.L[j].grad.data
            scn.L[j].data = (scn.L[j].data / (scn.L[j].data.sum())).clamp(0, 1)
            volatility*= 1.0
            #scn.L[j].data = torch.ones(scn.L[j].size()) / 2
        if i % 100 == 0:
            print(output)
            S.append(loss)
            print(loss)

        optimizer.zero_grad()

#visible_units = Variable(torch.FloatTensor([0, 1]).view(2, -1))
#scn = SCN(2, 1, visible_units, 1)

