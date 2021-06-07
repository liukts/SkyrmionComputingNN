import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from norse.torch.module import encode
from norse.torch import ConstantCurrentLIFEncoder

# load data, process into tensor
data = pd.read_csv("./wcancer_data.csv")
x = data.drop(["id", "diagnosis", "Unnamed: 32"], axis=1)  # prune unused data
diag = {"M": 1, "B": 0}
y = data["diagnosis"].replace(diag)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=85)
scaler = StandardScaler()
x_train_trans = scaler.fit_transform(x_train)
x_test_trans = scaler.fit_transform(x_test)
train = data_utils.TensorDataset(torch.from_numpy(x_train_trans).float(),
                                 torch.from_numpy(y_train.to_numpy()).float())
train_loader = data_utils.DataLoader(train, batch_size=128, shuffle=False)
data, label = train[100]
T = 100
example_encoder = encode.PoissonEncoder(seq_length=T,dt=0.001,f_max=1e2)
# example_encoder = ConstantCurrentLIFEncoder(seq_length=T)
example_input = example_encoder(data)
example_spikes = example_input.reshape(T,x_train.shape[1]).to_sparse().coalesce()
t = example_spikes.indices()[0]
n = example_spikes.indices()[1]

plt.scatter(t, n, marker='|', color='black')
plt.ylabel('Input Unit')
plt.xlabel('Time [0.1ns]')
plt.show()