#데이터 전처리

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        input_data = item['input']
        target_data = item['target']

        # Convert tuples to tensors
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        target_tensor = torch.tensor(target_data, dtype=torch.float32)

        return input_tensor, target_tensor


def separate_data(data):
    # 지역정보(code_new)별로 데이터 분리
    datas = {}
    for code in data['code_new'].unique():
        datas[code] = data[data['code_new'] == code].reset_index(drop=True)
    return datas

def data_interpolate(data):
    # 결측치 보간 로직 구현

    datas = separate_data(data)
    interp_datas = {}
    for code, df in datas.items():
        # ymd 기준으로 정렬 후 시간축 보간
        df_sorted = df.sort_values('ymd').reset_index(drop=True)
        # Set 'ymd' as index for time interpolation
        df_sorted = df_sorted.set_index('ymd')
        df_interp = df_sorted.interpolate(method='time', limit_direction='forward', axis=0)
        df_interp.reset_index(inplace=True)
        interp_datas[code] = df_interp

    interp_data = pd.concat(interp_datas.values(), ignore_index=True)
    # Sort by 'code_new' and then by the index (ymd)
    interp_data = interp_data.sort_values(['code_new']).sort_index().reset_index(drop=True)
    return interp_data



def make_dataset(file_path):
    data = pd.read_csv(file_path, encoding='cp949')
    data = data.rename(columns={'기온(°C)':'atemp','강수량(mm)':'precip','풍속(m/s)':'wspeed','습도(%)':'humid','현지기압(hPa)':'apress','지면온도(°C)':'gtemp'})
    data['ymd'] = pd.to_datetime(data['ymd'], errors='coerce')
    data = data_interpolate(data)

    # Normalization
    input_cols = ['elev', 'wtemp', 'ec', 'atemp', 'precip', 'wspeed', 'humid', 'apress', 'gtemp']
    scaler = MinMaxScaler()
    data[input_cols] = scaler.fit_transform(data[input_cols])

    #pd to list[dict]
    data_list = [
        {
            'input': (
                row.code_new,
                row.ymd.timestamp(),
                row.wtemp,
                row.ec,
                row.atemp,
                row.precip,
                row.wspeed,
                row.humid,
                row.apress,
                row.gtemp,
            ),
            'target': (
                row.elev
            )
        }
        for row in data.itertuples()
    ]
    dataset = CustomDataset(data_list)
    return dataset



tr_dataSet = make_dataset('/content/drive/MyDrive/2014_2020_시계열_지하수_기상_train.csv')
#ts_dataSet = make_dataset('/content/drive/MyDrive/2021_2023_시계열_지하수_기상_test_inputs.csv')

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



# GPU setting

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#LSTM 모델 정의
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
