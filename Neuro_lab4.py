import torch 
import torch.nn as nn 
import numpy as np
import pandas as pd

class NNet_regression(nn.Module):
    
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(nn.Linear(in_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, out_size)
                                    ) 
    def forward(self,X):
        pred = self.layers(X)
        return pred

df = pd.read_csv('d:\Download\dataset_simple.csv')

X = torch.Tensor(df.iloc[:, [0]].values)
print(X)
y = torch.Tensor(df.iloc[:, 1].values)

inputSize = X.shape[1]
hiddenSizes = 3
outputSize = 1

net = NNet_regression(inputSize,hiddenSizes,outputSize)
lossFn = nn.L1Loss()
optimizer = torch.optim.SGD(net.parameters(), lr=50)

epohs = 1000
for i in range(0,epohs):
    pred = net.forward(X)
    loss = lossFn(pred.squeeze(), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%1==0:
       print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())