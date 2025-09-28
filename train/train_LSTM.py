
#LSTM 베이스코드

import numpy as np

import pandas as pd

import pandas_datareader.data as pdr

import matplotlib.pyplot as plt

import datetime

import torch

import torch.nn as nn

from torch.autograd import Variable

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from train.Dataloader import LSTMDataset, make_dataset, get_dataloader
from utils.model.LSTM import LSTM

#데이터셋 불러오기
tr_dataSet = make_dataset('/content/drive/MyDrive/2014_2020_시계열_지하수_기상_train.csv')
ts_dataSet = make_dataset('/content/drive/MyDrive/2021_2023_시계열_지하수_기상_test_inputs.csv')

#dataloader
train_loader = get_dataloader(tr_dataSet, batch_size=2048, shuffle=True)
test_loader = get_dataloader(ts_dataSet, batch_size=2048, shuffle=False)

# GPU setting

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





#모델 학습 및 평가



num_epochs = 10000

learning_rate = 0.001

input_size = 10

hidden_size = 2 # number of features in hidden state

num_layers = 1

num_classes = 1

input_dataset = [item[0] for item in tr_dataSet]
target_dataset = [item[1] for item in tr_dataSet]

input_tensor = torch.stack(input_dataset)
target_tensor = torch.stack(target_dataset)

print(input_tensor.shape)
print(target_tensor.shape)

input_tensor = input_tensor.reshape(input_tensor.shape[0], 1, input_tensor.shape[1])
target_tensor = target_tensor.reshape(target_tensor.shape[0], 1, 1)

model = LSTM(num_classes, input_size, hidden_size, num_layers, input_tensor.shape[1]).to(device)



loss_function = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)



for epoch in range(num_epochs) :

    outputs = model(input_tensor.to(device))

    optimizer.zero_grad()

    loss = loss_function(outputs, target_tensor.to(device))

    loss.backward()

    optimizer.step() # improve from loss = back propagation

    if epoch % 200 == 0 :

        print("Epoch : %d, loss : %1.5f" % (epoch, loss.item()))



# Estimated Value

test_predict = model(input_tensor.to(device)) #Forward Pass

predict_data = test_predict.data.detach().cpu().numpy() #numpy conversion

predict_data = MinMaxScaler.inverse_transform(predict_data) #inverse normalization(Min/Max)



# Real Value

real_data = target_tensor.data.numpy() # Real value

real_data = MinMaxScaler.inverse_transform(real_data) #inverse normalization



#Figure

plt.figure(figsize = (10,6)) # Plotting

plt.plot(real_data, label = 'Real Data')

plt.plot(predict_data, label = 'predicted data')

plt.title('Time series prediction')

plt.legend()

plt.show()
