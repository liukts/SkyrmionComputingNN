import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np

from tqdm import tqdm, trange
import os

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# folder to save results
target_dir = "0628_wcancer_ann"

BATCH_SIZE = 500

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
train_loader = data_utils.DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)

test = data_utils.TensorDataset(torch.from_numpy(x_test_trans).float(),
                                 torch.from_numpy(y_test.to_numpy()).float())
test_loader = data_utils.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

INPUT_SIZE = x_train.shape[1]

class SeqNet(nn.Module):
    def __init__(self):
        super(SeqNet, self).__init__()
        self.l1 = nn.Linear(INPUT_SIZE, 2)
        self.r = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, INPUT_SIZE)
        x = self.r(self.l1(x))
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer):
    model.train()
    losses = []
    for (data, target) in tqdm(train_loader, desc='train', unit='batch', ncols=80, leave=False):
        data, target = data.to(device), target.long().to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    mean_loss = np.mean(losses)
    return losses, mean_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='test', unit='batch', ncols=80, leave=False):
            data, target = data.to(device), target.long().to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    return test_loss, accuracy

def save(path, epoch, model, optimizer, is_best=False):
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

rseed = 0
torch.manual_seed(rseed)

lr = 0.005
momentum = 0.1
model = SeqNet().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
epochs = 1500

train_losses = []
test_losses = []
mean_losses = []
accuracies = []
pbar = trange(epochs, ncols=80, unit="epoch")
for epoch in pbar:
    training_loss, mean_loss = train(model, DEVICE, train_loader, optimizer)
    test_loss, accuracy = test(model, DEVICE, test_loader)
    train_losses += training_loss
    mean_losses.append(mean_loss)
    test_losses.append(test_loss)
    accuracies.append(accuracy)       
    pbar.set_postfix(accuracy=accuracies[-1])

os.mkdir("./outputs/" + target_dir)
np.save("./outputs/" + target_dir + "/training_losses.npy", np.array(train_losses))
np.save("./outputs/" + target_dir + "/mean_losses.npy", np.array(mean_losses))
np.save("./outputs/" + target_dir + "/test_losses.npy", np.array(test_losses))
np.save("./outputs/" + target_dir + "/accuracies.npy", np.array(accuracies))
model_path = "./outputs/" + target_dir + "/model.pt"
save(
    model_path,
    epoch=epoch,
    model=model,
    optimizer=optimizer,
)
