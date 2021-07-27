import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tskone_module as tsk

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import numpy as np

from tqdm import tqdm, trange
import os

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# CONST
BATCH_SIZE = 500
EPOCHS = 800
LR = 0.05
SEEDS = 5

# FLAGS
ctext = True
direct = False

# folder to save results
target_dir = "0725_tskone_context"

if not os.path.isdir("./context/"):
    os.mkdir("./context/")
if not os.path.isdir("./context/" + target_dir):
    os.mkdir("./context/" + target_dir)

# load data, process into tensor
if ctext is False or direct is False:
    dataset = "./wcancer_data.csv"
    data = pd.read_csv(dataset)
    x = data.drop(["id", "diagnosis", "Unnamed: 32"], axis=1)  # prune unused data
    diag = {"M": 1, "B": 0}
    y = data["diagnosis"].replace(diag)
else:
    dataset = "./wcancer_data_context.csv"
    data = pd.read_csv(dataset)
    x = data.drop(["diagnosis"], axis=1) # remove class
    y = data["diagnosis"]
    check = {"T": 1, "F": 0}
    sex = {"F": 1, "M": 0}
    x['sex'] = x['sex'].replace(sex)
    x = x.replace(check)
context = torch.from_numpy(np.load('contextmask.npy')).to(DEVICE)
x_train, x_test, y_train, y_test, c_train, c_test = train_test_split(x, y, context, test_size=0.2,
                                                    random_state=85)
scaler = MinMaxScaler((3,4.5))
x_train_trans = scaler.fit_transform(x_train)
x_test_trans = scaler.fit_transform(x_test)
if direct is False:
    train = data_utils.TensorDataset(torch.from_numpy(x_train_trans).float(),
                                    torch.from_numpy(y_train.to_numpy()).float(),
                                    c_train.float())
    test = data_utils.TensorDataset(torch.from_numpy(x_test_trans).float(),
                                 torch.from_numpy(y_test.to_numpy()).float(),
                                 c_test.float())
else:
    train = data_utils.TensorDataset(torch.from_numpy(x_train_trans).float(),
                                    torch.from_numpy(y_train.to_numpy()).float())
    test = data_utils.TensorDataset(torch.from_numpy(x_test_trans).float(),
                                 torch.from_numpy(y_test.to_numpy()).float())
train_loader = data_utils.DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)
test_loader = data_utils.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

INPUT_SIZE = x_train.shape[1]

class SeqNet(nn.Module):
    def __init__(self):
        super(SeqNet, self).__init__()
        l1 = nn.Linear(INPUT_SIZE,2)
        self.t = tsk.TSKONE()
        self.tout = tsk.TSKONEout()
        self.l1_w = torch.nn.Parameter(l1.weight)
        self.l1_b = torch.nn.Parameter(l1.bias)

    def forward(self, x):
        x = x.view(-1, INPUT_SIZE)
        x = self.t(x,torch.ones_like(x))
        x = F.linear(x,self.l1_w,self.l1_b)
        x = torch.abs(x)
        return F.log_softmax(x, dim=1)

class SeqNetInf(nn.Module):
    def __init__(self):
        super(SeqNetInf, self).__init__()
        l1 = nn.Linear(INPUT_SIZE,2)
        self.t = tsk.TSKONE()
        self.tout = tsk.TSKONEout()
        self.l1_w = torch.nn.Parameter(l1.weight)
        self.l1_b = torch.nn.Parameter(l1.bias)

    def forward(self, x):
        x = x.view(-1, INPUT_SIZE)
        x = self.t(x,torch.ones_like(x))
        x = F.linear(x,self.l1_w,self.l1_b)
        x = self.tout(x)
        return F.log_softmax(x, dim=1)

class SeqNetCtext(nn.Module):
    def __init__(self):
        super(SeqNetCtext, self).__init__()
        l1 = nn.Linear(INPUT_SIZE,2)
        self.t = tsk.TSKONE()
        self.tout = tsk.TSKONEout()
        self.l1_w = torch.nn.Parameter(l1.weight)
        self.l1_b = torch.nn.Parameter(l1.bias)
        self.ct = tsk.CTEXTgen()

    def forward(self, x, ctext):
        x = x.view(-1, INPUT_SIZE)
        x,context = self.ct(x,ctext)
        x = self.t(x,context)
        x = F.linear(x,self.l1_w,self.l1_b)
        x = torch.abs(x)
        return F.log_softmax(x, dim=1)

class SeqNetCtextInf(nn.Module):
    def __init__(self):
        super(SeqNetCtextInf, self).__init__()
        l1 = nn.Linear(INPUT_SIZE,2)
        self.t = tsk.TSKONE()
        self.tout = tsk.TSKONEout()
        self.l1_w = torch.nn.Parameter(l1.weight)
        self.l1_b = torch.nn.Parameter(l1.bias)
        self.ct = tsk.CTEXTgen()

    def forward(self, x, ctext):
        x = x.view(-1, INPUT_SIZE)
        x,context = self.ct(x,ctext)
        x = self.t(x,context)
        x = F.linear(x,self.l1_w,self.l1_b)
        x = self.tout(x)
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, direct):
    model.train()
    losses = []
    if direct is False:
        for data, target, context in tqdm(train_loader, desc='train', unit='batch', ncols=80, leave=False):
            data, target, context = data.to(device), target.long().to(device), context.to(device)
            optimizer.zero_grad()
            output = model(data,context)
            loss = torch.nn.functional.nll_loss(output,target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    else:
        for data, target in tqdm(train_loader, desc='train', unit='batch', ncols=80, leave=False):
            data, target = data.to(device), target.long().to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.nll_loss(output,target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    mean_loss = np.mean(losses)
    weight = model.l1_w
    bias = model.l1_b
    return losses, mean_loss, weight, bias

def test(model, device, test_loader, direct, weight, bias):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        if direct is False:
            for data, target, context in tqdm(test_loader, desc='test', unit='batch', ncols=80, leave=False):
                data, target, context = data.to(device), target.long().to(device), context.to(device)
                model.l1_w.data = weight*0.13/(torch.abs(torch.max(weight)))
                model.l1_b.data = bias*0.13/(torch.abs(torch.max(weight)))
                output = model(data,context)
                test_loss += F.nll_loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        else:
            for data, target in tqdm(test_loader, desc='test', unit='batch', ncols=80, leave=False):
                data, target = data.to(device), target.long().to(device)
                model.l1_w.data = weight*0.13/(torch.abs(torch.max(weight)))
                model.l1_b.data = bias*0.13/(torch.abs(torch.max(weight)))
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

train_losses = np.empty((SEEDS,EPOCHS))
test_losses = np.empty((SEEDS,EPOCHS))
mean_losses = np.empty((SEEDS,EPOCHS))
accuracies = np.empty((SEEDS,EPOCHS))
for rseed in range(0,SEEDS):
    torch.manual_seed(rseed)
    if ctext is True and direct is False:
        model = SeqNetCtext().to(DEVICE)
        modeltest = SeqNetCtextInf().to(DEVICE)
    else:
        model = SeqNet().to(DEVICE)
        modeltest = SeqNetInf().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    pbar = trange(EPOCHS, ncols=80, unit="epoch")
    for epoch in pbar:
        training_loss, mean_loss, weight, bias = train(model, DEVICE, train_loader, optimizer, direct)
        test_loss, accuracy = test(modeltest, DEVICE, test_loader, direct, weight, bias)
        # train_losses[rseed,epoch] = training_loss
        mean_losses[rseed,epoch] = mean_loss
        test_losses[rseed,epoch] = test_loss
        accuracies[rseed,epoch] = accuracy
        pbar.set_postfix(accuracy=accuracy)

# np.save("./outputs/context/" + target_dir + "/train_losses.npy", np.array(train_losses))
np.save("./context/" + target_dir + "/mean_losses.npy", np.array(mean_losses))
np.save("./context/" + target_dir + "/test_losses.npy", np.array(test_losses))
np.save("./context/" + target_dir + "/accuracies.npy", np.array(accuracies))
model_path = "./context/" + target_dir + "/model.pt"
save(
    model_path,
    epoch=epoch,
    model=model,
    optimizer=optimizer,
)
