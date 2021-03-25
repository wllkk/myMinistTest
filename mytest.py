import torch
import sys
a = torch.randn(10)
b = torch.randn(10)
c = torch.randn(10)
print(a)
bn = torch.zeros(a.shape[0] * 3)
print(bn)

i = 0
index = 0

for x in range(3):
    size = a.shape[0]
    bn[index:(index+size)] = a.data.abs()
    index+=size
j, k = torch.sort(bn)
print(bn)
print(j)
print(k)


x = torch.randn(10)
y = 0.2
print(x.abs())
mask = x.abs().gt(y).float()
print(mask)
remin_channel = torch.sum(mask)
print(remin_channel)

z = int(torch.argmax(x))
print("z",z)