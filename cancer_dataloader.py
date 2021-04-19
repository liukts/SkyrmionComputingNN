# Import Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import tarfile

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torchvision import transforms
from collections import Counter

# Save Train, Validation and Test Data
def save_data(data, mode="train"):
    i = 0
    for img, label in data:
        folder_path = os.path.join(os.path.join(os.getcwd(), mode), str(np.array(label)[0]))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        i += 1
        image = transforms.ToPILImage()(img[0, :, :, :])
        image.save(os.path.join(folder_path, mode+"_"+str(i)+".jpg"), "JPEG")

project_name='breast-cancer-data'

# Load all image data
data_dir = os.getcwd()
folder_name = "cancer"
image_folders = os.path.join(data_dir, folder_name)

transform = transforms.Compose([transforms.Resize((50, 50)), transforms.ToTensor()])
images = []
for file in os.listdir(image_folders):
    images.append(ImageFolder(os.path.join(image_folders, file), transform=transform))
datasets = torch.utils.data.ConcatDataset(images)

# Determine the number of samples for each class
i = 0
for dataset in datasets.datasets:
    if i == 0:
        result = Counter(dataset.targets)
        i += 1
    else:
        result += Counter(dataset.targets)

result = dict(result)
print("""Total Number of Images for each Class:
    Class 0 (No Breast Cancer): {}
    Class 1 (Breast Cancer present): {}""".format(result[0], result[1]))

# Prepare data for training, validation and test
random_seed = 52893
torch.manual_seed(random_seed)

test_size = 38000
train_size = len(datasets) - test_size
train_ds, test_ds = random_split(datasets, [train_size, test_size])

val_size = 38000
train_size = len(train_ds) - val_size
train_ds, val_ds = random_split(train_ds, [train_size, val_size])

len(train_ds), len(val_ds), len(test_ds)

train_data = DataLoader(train_ds, shuffle=True, num_workers=4, pin_memory=True)
val_data = DataLoader(val_ds, shuffle=True, num_workers=4, pin_memory=True)
test_data = DataLoader(test_ds, shuffle=True, num_workers=4, pin_memory=True)

# Save Train Data
save_data(train_data, mode="train")

# Save Validation Data
save_data(val_data, mode="validation")

# Save Test Data
save_data(test_data, mode="test")