import torch
from torch._C import device
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm, trange
import os

# folder to save results
target_dir = "0614_both_SeqNet"

BATCH_SIZE = 100
INPUT_SIZE = 28 * 28
MODEL_TYPE = "SeqNet" # choices are ConvNet or SeqNet
DATASET = "both" # choices are mnist, kmnist, or both

mnist_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

kmnist_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ]
)

train_data_mnist = torchvision.datasets.MNIST(
    root=".",
    train=True,
    download=True,
    transform=mnist_transform,
)
test_data_mnist = torchvision.datasets.MNIST(
    root=".",
    train=False,
    transform=mnist_transform,
)
train_loader_mnist = torch.utils.data.DataLoader(
    train_data_mnist,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader_mnist = torch.utils.data.DataLoader(
    test_data_mnist,
    batch_size=BATCH_SIZE
)

train_data_kmnist = torchvision.datasets.KMNIST(
    root=".",
    train=True,
    download=True,
    transform=kmnist_transform,
)
test_data_kmnist = torchvision.datasets.KMNIST(
    root=".",
    train=False,
    transform=kmnist_transform,
)
train_loader_kmnist = torch.utils.data.DataLoader(
    train_data_kmnist,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader_kmnist = torch.utils.data.DataLoader(
    test_data_kmnist,
    batch_size=BATCH_SIZE
)

train_data_dual = torch.utils.data.ConcatDataset(
    (train_data_mnist, train_data_kmnist)
)
test_data_dual = torch.utils.data.ConcatDataset(
    (test_data_mnist, test_data_kmnist)
)
train_loader_dual = torch.utils.data.DataLoader(
    train_data_dual,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader_dual = torch.utils.data.DataLoader(
    test_data_dual,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class SeqNet(nn.Module):
    def __init__(self):
        super(SeqNet, self).__init__()
        self.l1 = nn.Linear(INPUT_SIZE, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 10)
        self.r = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, INPUT_SIZE)
        x = self.r(self.l1(x))
        x = self.r(self.l2(x))
        x = self.l3(x)
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer):
    model.train()
    losses = []
    for (data, target) in tqdm(train_loader, desc='train', unit='batch', ncols=80, leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
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
            data, target = data.to(device), target.to(device)
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

if DATASET == "mnist":
    train_loader = train_loader_mnist
    test_loader = test_loader_mnist
elif DATASET == "kmnist":
    train_loader = train_loader_kmnist
    test_loader = test_loader_kmnist
elif DATASET == "both":
    train_loader = train_loader_dual
    test_loader = test_loader_dual

rseed = 0
torch.manual_seed(rseed)
if MODEL_TYPE == "ConvNet":
    lr = 0.005
    momentum = 0.9
    model = ConvNet().to(DEVICE)
elif MODEL_TYPE == "SeqNet":
    lr = 0.003
    momentum = 0.9
    model = SeqNet().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
epochs = 15

train_losses = []
test_losses = []
mean_losses = []
accuracies = []
if DATASET == "both":
    mnist_accuracies = []
    kmnist_accuracies = []
pbar = trange(epochs, ncols=80, unit="epoch")
for epoch in pbar:
    training_loss, mean_loss = train(model, DEVICE, train_loader, optimizer)
    test_loss, accuracy = test(model, DEVICE, test_loader)
    if DATASET == "both":
        test_loss_mnist, mnist_accuracy = test(model, DEVICE, test_loader_mnist)
        test_loss_kmnist, kmnist_accuracy = test(model, DEVICE, test_loader_kmnist)
        mnist_accuracies.append(mnist_accuracy)
        kmnist_accuracies.append(kmnist_accuracy)
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
if DATASET == "both":
    np.save("./outputs/" + target_dir + "/mnist_accuracies.npy", np.array(mnist_accuracies))
    np.save("./outputs/" + target_dir + "/kmnist_accuracies.npy", np.array(kmnist_accuracies))
model_path = "./outputs/" + target_dir + "/model.pt"
save(
    model_path,
    epoch=epoch,
    model=model,
    optimizer=optimizer,
)
