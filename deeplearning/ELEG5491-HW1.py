import torch
from torch import nn

x1 = torch.tensor([-0.7411, -0.5078, -0.3206]).reshape(3, 1)
x2 = torch.tensor([0.0983, -0.0308, -0.3728]).reshape(3, 1)
x3 = torch.tensor([0.0414, 0.2323, -0.2365]).reshape(3, 1)

'''
X1 = torch.tensor([[-0.7411, -0.5078, -0.3206],
                   [0.0983, -0.0308, -0.3728],
                   [0.0414, 0.2323, -0.2365],
                   [-0.7342, 0.4264, 2.0237]])
X1 = X1.T
Y = torch.tensor([[0, 1, 0],
                  [1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
Y = Y.T
'''
target = torch.tensor([1, 0, 1, 2])

t1 = torch.tensor([1])
t2 = torch.tensor([0])
t3 = torch.tensor([1])


W1 = torch.tensor([[1.6035, -1.5062, 0.2761],
                   [1.2347, -0.4446, -0.2612],
                   [-0.2296, -0.1559, 0.4434]], requires_grad=True)
b1 = torch.tensor([0.3919, -1.2507, -0.9480], requires_grad=True).reshape(3, 1)
W2 = torch.tensor([[0.0125, 1.2424, 0.3503],
                   [- 3.0292, -1.0667, -0.0290],
                   [- 0.4570, 0.9337, 0.1825]], requires_grad=True)
b2 = torch.tensor([-1.5651, -0.0845, 1.6039], requires_grad=True).reshape(3, 1)
b2r = torch.tensor([-1.5654, -0.0760, 1.5957], requires_grad=True).reshape(3, 1)

'''
L1 = torch.matmul(W1, X1) + b1
print('L1:', L1)
a = torch.zeros_like(L1)
relu = torch.max(L1, a)
print('relu:', relu)
L2 = torch.matmul(W2, relu) + b2
print('L2:', L2)
softmax_func = nn.Softmax(dim=0)
softmax = softmax_func(L2)
print('softmax:', softmax)
loss = nn.CrossEntropyLoss()
print('loss:', loss(L2.T, target))
print('\n')
'''

L1 = torch.matmul(W1, x1) + b1
print('1-L1',L1)
a = torch.zeros_like(L1)
relu = torch.max(L1, a)
print('1-relu',relu)
L2 = torch.matmul(W2, relu) + b2
print('1-L2:',L2)
softmax_func = nn.Softmax(dim=0)
softmax = softmax_func(L2)
print('1-softmax:', softmax)
loss = nn.CrossEntropyLoss()
print('1-loss:', loss(L2.T, t1))
print('\n \n')

L1 = torch.matmul(W1, x3) + b1
print('3-L1', L1)
a = torch.zeros_like(L1)
relu = torch.max(L1, a)
print('3-relu',relu)
L2 = torch.matmul(W2, relu) + b2r
print('3-L2:',L2)
softmax_func = nn.Softmax(dim=0)
softmax = softmax_func(L2)
print('3-softmax:', softmax)
loss = nn.CrossEntropyLoss()
print('loss:', loss(L2.T, t1), '\n\n')

L1 = torch.matmul(W1, x2) + b1
print('2-L1', L1)
a = torch.zeros_like(L1)
relu = torch.max(L1, a)
print('2-relu',relu)
L2 = torch.matmul(W2, relu) + b2r
print('2-L2:',L2)
softmax_func = nn.Softmax(dim=0)
softmax = softmax_func(L2)
print('2-softmax:', softmax)
loss = nn.CrossEntropyLoss()
print('loss:', loss(L2.T, t1))



"""
X2p = torch.tensor([[0.2642, 0.3336, 0.4023],
                    [0.3995, 0.3511, 0.2494],
                    [0.3370, 0.4078, 0.2552],
                    [0.0501, 0.1599, 0.7900]])

X2 = torch.transpose(X2p, 0, 1)
"""
