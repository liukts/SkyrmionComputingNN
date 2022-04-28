import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler

import numpy as np

from tqdm import tqdm, trange
import os

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# folder to save results
target_dir = "220428_wcancer_ann_direct"

direct = True
BATCH_SIZE = 1000

# load data, process into tensor
if direct:
    dataset = "./wcancer_data_context.csv"
    data = pd.read_csv(dataset)
    x = data.drop(["diagnosis"], axis=1) # remove class
    y = data["diagnosis"]
    check = {"T": 1, "F": 0}
    sex = {"F": 1, "M": 0}
    x['sex'] = x['sex'].replace(sex)
    x = x.replace(check)
else:
    data = pd.read_csv("./wcancer_data.csv")
    x = data.drop(["id", "diagnosis", "Unnamed: 32"], axis=1)  # prune unused data
    diag = {"M": 1, "B": 0}
    y = data["diagnosis"].replace(diag)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=85)
# scaler = MinMaxScaler((3,4.5))
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
        self.l1 = nn.Linear(INPUT_SIZE, 2, bias=False)
        self.r = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, INPUT_SIZE)
        x = self.l1(x)
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer):
    model.train()
    losses = []
    correct = 0
    for (data, target) in tqdm(train_loader, desc='train', unit='batch', ncols=120, leave=False):
        data, target = data.to(device), target.long().to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss = torch.nn.functional.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    mean_loss = np.mean(losses)
    train_accuracy = 100.0 * correct / len(train_loader.dataset)
    return losses, mean_loss, train_accuracy

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='test', unit='batch', ncols=120, leave=False):
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

lr = 0.1
epochs = 500
seeds = 5

train_losses = []
test_losses = []
mean_losses = []
accuracies = []
train_accs = []
for i in range(seeds):
    torch.manual_seed(i)
    model = SeqNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    pbar = trange(epochs, ncols=120, unit="epoch")
    for epoch in pbar:
        training_loss, mean_loss, train_acc = train(model, DEVICE, train_loader, optimizer)
        test_loss, accuracy = test(model, DEVICE, test_loader)
        train_losses += training_loss
        mean_losses.append(mean_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)       
        train_accs.append(train_acc)
        pbar.set_postfix(accuracy=(train_accs[-1],accuracies[-1]))
train_losses = np.array(train_losses).reshape(seeds,epochs)
mean_losses = np.array(mean_losses).reshape(seeds,epochs)
test_losses = np.array(test_losses).reshape(seeds,epochs)
accuracies = np.array(accuracies).reshape(seeds,epochs)
train_accs = np.array(train_accs).reshape(seeds,epochs)
os.mkdir("./context/" + target_dir)
np.save("./context/" + target_dir + "/training_losses.npy",train_losses)
np.save("./context/" + target_dir + "/mean_losses.npy", mean_losses)
np.save("./context/" + target_dir + "/test_losses.npy", test_losses)
np.save("./context/" + target_dir + "/accuracies.npy", accuracies)
np.save("./context/" + target_dir + "/train_accs.npy", train_accs)
model_path = "./context/" + target_dir + "/model.pt"
save(
    model_path,
    epoch=epoch,
    model=model,
    optimizer=optimizer,
)
