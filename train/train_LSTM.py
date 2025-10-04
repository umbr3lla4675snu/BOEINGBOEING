#필요한 헤더 import

import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler


#데이터셋 불러오기


#from train.Dataloader import LSTMDataset, make_dataset, get_dataloader
#from utils.model.LSTM import LSTM

tr_dataSet = make_dataset('/content/drive/MyDrive/2014_2020_시계열_지하수_기상_train.csv')

trset_size = len(tr_dataSet)

val_ratio = 0.3
val_size = int(val_ratio * len(tr_dataSet))
tr_size = len(tr_dataSet) - val_size

train_dataset, val_dataset = torch.utils.data.random_split(tr_dataSet, [tr_size, val_size])

train_loader = get_dataloader(train_dataset, batch_size=2048, shuffle=True)
val_loader = get_dataloader(val_dataset, batch_size=2048, shuffle=False)

#LSTM 모델 구성


class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length) :
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
        self.layer_1 = nn.Linear(hidden_size, 256)
        self.layer_2 = nn.Linear(256,256)
        self.layer_3 = nn.Linear(256,128)
        self.layer_out = nn.Linear(128, num_classes)
        self.relu = nn.ReLU() #Activation Func

    def forward(self,x):
        # Reshape input to be (batch_size, seq_length, input_size)
        x = x.view(x.size(0), self.seq_length, self.input_size)

        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #Hidden State
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #Internal Process States

        output, (hn, cn) = self.lstm(x, (h_0, c_0))

        hn = hn.view(-1, self.hidden_size) # Reshaping the data for starting LSTM network
        out = self.relu(hn) #pre-processing for first layer
        out = self.layer_1(out) # first layer
        out = self.relu(out) # activation func relu
        out = self.layer_2(out)
        out = self.relu(out)
        out = self.layer_3(out)
        out = self.relu(out)
        out = self.layer_out(out) #Output layer
        return out
    

    #LSTM 학습

#test_loader = get_dataloader(ts_dataSet, batch_size=2048, shuffle=False)

# GPU setting

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#모델 학습 및 평가
num_epochs = 10000

learning_rate = 0.001

input_size = 10 # Number of features in the input tensor

hidden_size = 2 # number of features in hidden state

num_layers = 1

seq_length = 1000 # Should match the actual sequence length from the Dataset (2 * seq_day)

num_classes = 1


#input_dataset = [item[0] for item in tr_dataSet]
#target_dataset = [item[1] for item in tr_dataSet]

#input_tensor = torch.stack(input_dataset)
#target_tensor = torch.stack(target_dataset)

#print(input_tensor.shape)
#print(target_tensor.shape)

#input_tensor = input_tensor.reshape(input_tensor.shape[0], 1, input_tensor.shape[1])
#target_tensor = target_tensor.reshape(target_tensor.shape[0], 1, 1)

model = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length).to(device)



loss_function = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Learning Rate Scheduler 추가
from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(optimizer, step_size=500, gamma=0.1) # 500 에포크마다 learning rate를 0.1배 감소


for epoch in range(num_epochs) :

    rloss = 0.0;

    for inputs, targets in train_loader:

        # Inputs from DataLoader are (batch_size, actual_seq_length, 1, input_size)
        # Targets from DataLoader are (batch_size, actual_seq_length, 1, num_classes)
        # Reshape inputs to be (batch_size, seq_length, input_size)
        inputs = inputs.view(inputs.size(0), seq_length, input_size).to(device)
        # Reshape targets to be (batch_size, seq_length, num_classes)
        targets = targets.view(targets.size(0), seq_length, num_classes).to(device)


        optimizer.zero_grad()

        outputs = model(inputs)

        # Reshape outputs to match targets for loss calculation
        # outputs are (batch_size, num_classes) from the last layer of LSTM
        # targets need to be (batch_size, num_classes) from the last step of the sequence
        outputs = outputs.view(outputs.size(0), num_classes)
        targets = targets[:, -1, :].view(targets.size(0), num_classes)


        loss = loss_function(outputs, targets)

        print("Epoch : %d, loss : %1.5f" % (epoch, loss.item()))

        loss.backward()
        optimizer.step()

    # 스케줄러 스텝 (에포크마다 호출)
    scheduler.step()


# Estimated Value

#test_predict = model(input_tensor.to(device)) #Forward Pass

#predict_data = test_predict.data.detach().cpu().numpy() #numpy conversion

#predict_data = MinMaxScaler.inverse_transform(predict_data) #inverse normalization(Min/Max)



# Real Value

#real_data = target_tensor.data.numpy() # Real value

#real_data = MinMaxScaler.inverse_transform(real_data) #inverse normalization



#Figure

#plt.figure(figsize = (10,6)) # Plotting

#plt.plot(real_data, label = 'Real Data')

#plt.plot(predict_data, label = 'predicted data')

#plt.title('Time series prediction')

#plt.legend()

#plt.show()



# Estimated Value and Visualization using DataLoader

# 모델을 평가 모드로 설정
model.eval()

# val_loader에서 모든 데이터 가져오기
all_inputs = []
all_targets = []
with torch.no_grad():
    for inputs, targets in val_loader:
        # Reshape inputs to be (batch_size, seq_length, input_size)
        inputs = inputs.view(inputs.size(0), seq_length, input_size)
        # Reshape targets to be (batch_size, seq_length, num_classes)
        targets = targets.view(targets.size(0), seq_length, num_classes)

        all_inputs.append(inputs)
        all_targets.append(targets)

# 모든 배치의 데이터를 하나의 텐서로 합치기
all_inputs = torch.cat(all_inputs, dim=0).to(device)
all_targets = torch.cat(all_targets, dim=0).to(device)


# 전체 검증 데이터에 대한 모델 예측
test_predict = model(all_inputs) #Forward Pass

# 예측 결과와 실제 목표값 가져오기 (시퀀스의 마지막 값 사용)
predict_data_scaled = test_predict.data.detach().cpu().numpy() # numpy conversion
real_data_scaled = all_targets[:, -1, :].data.detach().cpu().numpy() # Real value from the last step


# Inverse transform the predicted and actual values
# Create dummy arrays to match the shape expected by the scaler's inverse_transform
# The scaler was fitted on ['elev', 'wtemp', 'ec', 'atemp', 'precip', 'wspeed', 'humid', 'apress', 'gtemp']
# 'elev' is the first column in the input_cols list used for fitting
dummy_predicted = np.zeros((predict_data_scaled.shape[0], len(input_cols)))
dummy_predicted[:, 0] = predict_data_scaled.flatten() # Place predicted 'elev' values in the first column

dummy_actual = np.zeros((real_data_scaled.shape[0], len(input_cols)))
dummy_actual[:, 0] = real_data_scaled.flatten() # Place actual 'elev' values in the first column

# Ensure scaler is fitted. If make_dataset was executed, it should be.
# Note: make_dataset should be executed before this cell to define and fit the scaler
if not hasattr(scaler, 'scale_'):
    print("경고: Scaler가 아직 fit되지 않았습니다. make_dataset 함수가 실행되었는지 확인하세요.")
    # Handle this case, e.g., by re-running make_dataset or skipping inverse transform


predict_data = scaler.inverse_transform(dummy_predicted)[:, 0] # inverse normalization(Min/Max)
real_data = scaler.inverse_transform(dummy_actual)[:, 0] # inverse normalization


#Figure

plt.figure(figsize = (10,6)) # Plotting

plt.plot(real_data, label = 'Real Data')

plt.plot(predict_data, label = 'predicted data')

plt.title('Time series prediction (Validation Set)')

plt.legend()

plt.show()