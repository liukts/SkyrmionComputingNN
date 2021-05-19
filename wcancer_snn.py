import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator

import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
import torch.nn.init as init

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
                                 torch.from_numpy(y_train.as_matrix()).float())
dataloader = data_utils.DataLoader(train, batch_size=128, shuffle=False)
