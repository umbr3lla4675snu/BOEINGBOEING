import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class LSTMDataset(Dataset):
    def __init__(self, data_list, seq_day=500):
        self.data_list = data_list
        self.data_seq = []
        for i in range(0, len(self.data_list) - 2*seq_day, seq_day):
            if self.data_list[i]['input'][0] != self.data_list[i+2*seq_day-1]['input'][0]:
                continue
            else:
                self.data_seq.append(self.data_list[i:i+2*seq_day])

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        item = self.data_seq[idx]
        input_data = [torch.tensor(d['input'], dtype=torch.float32) for d in item]
        target_data = [torch.tensor(d['target'], dtype=torch.float32) for d in item]

        # Convert tuples to tensors
        #input_tensor = {seq_day, 1, input_dim}
        #target_tensor = {seq_day, 1, target_dim = 1}
        input_tensor = torch.stack(input_data).reshape(-1, 1, len(input_data[0]))
        target_tensor = torch.stack(target_data).reshape(-1, 1, 1)

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
    dataset = LSTMDataset(data_list, seq_day=500)
    return dataset

def get_dataloader(dataset, batch_size=2048, shuffle=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader