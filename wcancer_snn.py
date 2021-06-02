import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data_utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from norse.torch import LIFParameters, LIFState
from norse.torch.module.lif import LIFRecurrentCell
from norse.torch import LICell, LIState, ConstantCurrentLIFEncoder
from norse.torch.module import encode

from tqdm import tqdm, trange
from typing import NamedTuple

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

BATCH_SIZE = 100

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
train = data_utils.TensorDataset(torch.from_numpy(x_train_trans),
                                 torch.from_numpy(y_train.to_numpy()))
train_loader = data_utils.DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)

test = data_utils.TensorDataset(torch.from_numpy(x_test_trans),
                                 torch.from_numpy(y_test.to_numpy()))
test_loader = data_utils.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

class SNNState(NamedTuple):
    lif0 : LIFState
    readout : LIState


class SNN(torch.nn.Module):
    def __init__(self, input_features, hidden_features, output_features, record=False, dt=0.001):
        super(SNN, self).__init__()
        self.l1 = LIFRecurrentCell(
            input_features,
            hidden_features,
            p=LIFParameters(method='super',alpha=100),
            dt=dt             
        )
        self.input_features = input_features
        self.fc_out = torch.nn.Linear(hidden_features, output_features, bias=False)
        self.out = LICell(dt=dt)

        self.hidden_features = hidden_features
        self.output_features = output_features
        self.record = record

    def forward(self, x):
        seq_length, batch_size, _ = x.shape
        s1 = so = None
        voltages = []

        for ts in range(seq_length):
            z = x[ts, :, :].view(-1, self.input_features)
            z, s1 = self.l1(z, s1)
            z = self.fc_out(z)
            vo, so = self.out(z, so)
            voltages += [vo]

        return torch.stack(voltages)

class Model(torch.nn.Module):
    def __init__(self, encoder, snn, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.snn = snn
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.snn(x)
        log_p_y = self.decoder(x)
        return log_p_y

def decode(x):
    x, _ = torch.max(x, 0)
    log_p_y = torch.nn.functional.log_softmax(x, dim=1)
    return log_p_y

def train(model, device, train_loader, optimizer, epoch, max_epochs):
    model.train()
    losses = []

    for (data, target) in tqdm(train_loader, leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    mean_loss = np.mean(losses)
    return losses, mean_loss

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(
                output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100.0 * correct / len(test_loader.dataset)

    return test_loss, accuracy

T = 70
LR = 0.01
INPUT_FEATURES = x_train.shape[1]
HIDDEN_FEATURES = 100
OUTPUT_FEATURES = 2
EPOCHS = 50

model = Model(
    encoder=encode.PoissonEncoder(seq_length=T),
    snn=SNN(
      input_features=INPUT_FEATURES,
      hidden_features=HIDDEN_FEATURES,
      output_features=OUTPUT_FEATURES
    ),
    decoder=decode
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

training_losses = []
mean_losses = []
test_losses = []
accuracies = []

torch.autograd.set_detect_anomaly(True)

pbar = trange(EPOCHS, unit="epoch")
for epoch in pbar:
    training_loss, mean_loss = train(model, DEVICE, train_loader, optimizer, epoch, max_epochs=EPOCHS)
    test_loss, accuracy = test(model, DEVICE, test_loader, epoch)
    training_losses += training_loss
    mean_losses.append(mean_loss)
    test_losses.append(test_loss)
    accuracies.append(accuracy)
    pbar.set_postfix(accuracy=accuracies[-1])

print(f"final accuracy: {accuracies[-1]}")