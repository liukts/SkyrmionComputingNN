import numpy as np
import torch
import torchvision
import torch.utils.data as data_utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tskone_module as tsk
import numpy as np
from tqdm import tqdm, trange
import os

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# folder to save results
target_dir = "220719_mnist_ann_base_lr0.01"

direct = True
BATCH_SIZE = 300
INPUT_SIZE = 28*28
lr = 0.01
epochs = 15
seeds = 5

# load data, process into tensor
train_data = torchvision.datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = torchvision.transforms.ToTensor(), 
    download = True,            
)
test_data = torchvision.datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = torchvision.transforms.ToTensor()
)
train_loader = data_utils.DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
test_loader = data_utils.DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True)

class SeqNet(nn.Module):
    def __init__(self):
        super(SeqNet, self).__init__()
        self.l1 = nn.Linear(INPUT_SIZE, 10, bias=True)
        self.l2 = nn.Linear(200,10)
        self.r = nn.ReLU()
        self.t = tsk.TSKONE()

    def forward(self, x, targ):
        x = x.view(-1, INPUT_SIZE)
        x = x*1.5 + 3
        # x = self.t(x,targ.repeat(INPUT_SIZE,1).T % 2)
        x = self.t(x,torch.zeros_like(x))
        x = self.l1(x)
        x = torch.abs(x)
        return F.log_softmax(x, dim=1)

class SeqNetDirect(nn.Module):
    def __init__(self):
        super(SeqNetDirect, self).__init__()
        self.l1 = nn.Linear(INPUT_SIZE+1, 10, bias=False)
        self.r = nn.ReLU()
        self.t = tsk.TSKONE()

    def forward(self, x, targ):
        x = x.view(-1, INPUT_SIZE)
        x = x*1.5 + 3
        # x = torch.cat((x, targ.reshape(-1,1) % 2),1)
        x = torch.cat((x, torch.zeros_like(targ).reshape(-1,1)),1)
        x = self.t(x,targ.repeat(INPUT_SIZE+1,1).T % 2)
        # x = self.t(x,torch.zeros_like(x))
        x = self.l1(x)
        x = torch.abs(x)
        return F.log_softmax(x, dim=1)

class SeqNetReg(nn.Module):
    def __init__(self):
        super(SeqNetReg, self).__init__()
        self.l1 = nn.Linear(INPUT_SIZE, 10, bias=True)
        self.r = nn.ReLU()
        self.t = tsk.TSKONE()

    def forward(self, x, targ):
        x = x.view(-1, INPUT_SIZE)
        # x = self.t(x,targ.repeat(INPUT_SIZE,1).T % 2)
        # x = self.t(x,torch.zeros_like(x))
        x = self.l1(x)
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer):
    model.train()
    losses = []
    correct = 0
    for (data, target) in tqdm(train_loader, desc='train', unit='batch', ncols=120, leave=False):
        data, target = data.to(device), target.long().to(device)
        optimizer.zero_grad()
        output = model(data,target)
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
            output = model(data,target)
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

train_losses = []
test_losses = []
mean_losses = []
accuracies = []
train_accs = []
for i in range(seeds):
    torch.manual_seed(i)
    model = SeqNetReg().to(DEVICE)
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
# train_losses = np.array(train_losses).reshape(seeds,epochs)
mean_losses = np.array(mean_losses).reshape(seeds,epochs)
test_losses = np.array(test_losses).reshape(seeds,epochs)
accuracies = np.array(accuracies).reshape(seeds,epochs)
# train_accs = np.array(train_accs).reshape(seeds,epochs)
if not os.path.isdir("./context/" + target_dir):
    os.mkdir("./context/" + target_dir)
#np.save("./context/" + target_dir + "/training_losses.npy",train_losses)
np.save("./context/" + target_dir + "/mean_losses.npy", mean_losses)
np.save("./context/" + target_dir + "/test_losses.npy", test_losses)
np.save("./context/" + target_dir + "/accuracies.npy", accuracies)
#np.save("./context/" + target_dir + "/train_accs.npy", train_accs)
model_path = "./context/" + target_dir + "/model.pt"
save(
    model_path,
    epoch=epoch,
    model=model,
    optimizer=optimizer,
)
