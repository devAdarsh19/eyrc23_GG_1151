# %% [code] {"execution":{"iopub.status.busy":"2023-10-23T09:11:18.755421Z","iopub.execute_input":"2023-10-23T09:11:18.755713Z","iopub.status.idle":"2023-10-23T09:11:23.290715Z","shell.execute_reply.started":"2023-10-23T09:11:18.755664Z","shell.execute_reply":"2023-10-23T09:11:23.289671Z"}}
import random
import torch
from PIL import Image
import torchvision
import sklearn
import os
import PIL
import torchvision.transforms as transforms
import glob
from torch.utils.data import DataLoader
import pathlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim

# %% [code] {"execution":{"iopub.status.busy":"2023-10-23T09:11:23.292598Z","iopub.execute_input":"2023-10-23T09:11:23.293116Z","iopub.status.idle":"2023-10-23T09:11:23.307864Z","shell.execute_reply.started":"2023-10-23T09:11:23.293081Z","shell.execute_reply":"2023-10-23T09:11:23.307009Z"}}
os.listdir('/kaggle/input/war-project-zip/war/train')

# %% [code] {"execution":{"iopub.status.busy":"2023-10-23T09:12:55.788365Z","iopub.execute_input":"2023-10-23T09:12:55.788734Z","iopub.status.idle":"2023-10-23T09:12:55.792971Z","shell.execute_reply.started":"2023-10-23T09:12:55.788704Z","shell.execute_reply":"2023-10-23T09:12:55.792059Z"}}
path = '/kaggle/working/'


# %% [code] {"execution":{"iopub.status.busy":"2023-10-23T09:11:23.308965Z","iopub.execute_input":"2023-10-23T09:11:23.309294Z","iopub.status.idle":"2023-10-23T09:11:23.320459Z","shell.execute_reply.started":"2023-10-23T09:11:23.309261Z","shell.execute_reply":"2023-10-23T09:11:23.319691Z"}}
def get_mean_and_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0
    for images, _ in loader:
        image_count_in_a_batch = images.size(0)
        images = images.view(image_count_in_a_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

        total_images_count += image_count_in_a_batch

    mean /= total_images_count
    std /= total_images_count

    return mean, std


# %% [code] {"execution":{"iopub.status.busy":"2023-10-23T09:11:23.322933Z","iopub.execute_input":"2023-10-23T09:11:23.323508Z","iopub.status.idle":"2023-10-23T09:11:23.344811Z","shell.execute_reply.started":"2023-10-23T09:11:23.323475Z","shell.execute_reply":"2023-10-23T09:11:23.344042Z"}}
mean = [0.4873, 0.4838, 0.4746]
std = [0.2585, 0.2584, 0.2712]

train_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    # transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

test_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

# %% [code] {"execution":{"iopub.status.busy":"2023-10-23T09:11:23.345814Z","iopub.execute_input":"2023-10-23T09:11:23.346077Z","iopub.status.idle":"2023-10-23T09:11:23.350113Z","shell.execute_reply.started":"2023-10-23T09:11:23.346055Z","shell.execute_reply":"2023-10-23T09:11:23.349252Z"}}
train_dataset_path = '/kaggle/input/war-project-zip/war/train'
test_dataset_path = '/kaggle/input/war-project-zip/war/valid'

# %% [code] {"execution":{"iopub.status.busy":"2023-10-23T09:11:23.351142Z","iopub.execute_input":"2023-10-23T09:11:23.351924Z","iopub.status.idle":"2023-10-23T09:11:24.372624Z","shell.execute_reply.started":"2023-10-23T09:11:23.351893Z","shell.execute_reply":"2023-10-23T09:11:24.371721Z"}}
train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=train_transform)

test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=test_transform)

# %% [code] {"execution":{"iopub.status.busy":"2023-10-23T09:11:24.373690Z","iopub.execute_input":"2023-10-23T09:11:24.373964Z","iopub.status.idle":"2023-10-23T09:11:24.380512Z","shell.execute_reply.started":"2023-10-23T09:11:24.373941Z","shell.execute_reply":"2023-10-23T09:11:24.379541Z"}}
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)


# %% [code] {"execution":{"iopub.status.busy":"2023-10-23T09:11:24.381548Z","iopub.execute_input":"2023-10-23T09:11:24.381832Z","iopub.status.idle":"2023-10-23T09:11:24.393546Z","shell.execute_reply.started":"2023-10-23T09:11:24.381808Z","shell.execute_reply":"2023-10-23T09:11:24.392720Z"}}
def get_mean_and_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0
    for images, _ in loader:
        image_count_in_a_batch = images.size(0)
        images = images.view(image_count_in_a_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

        total_images_count += image_count_in_a_batch

    mean /= total_images_count
    std /= total_images_count

    return mean, std


# %% [code] {"execution":{"iopub.status.busy":"2023-10-23T09:11:24.394639Z","iopub.execute_input":"2023-10-23T09:11:24.394924Z","iopub.status.idle":"2023-10-23T09:11:24.413379Z","shell.execute_reply.started":"2023-10-23T09:11:24.394901Z","shell.execute_reply":"2023-10-23T09:11:24.412635Z"}}
root = pathlib.Path(train_dataset_path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
classes


# %% [code] {"execution":{"iopub.status.busy":"2023-10-23T09:11:24.416305Z","iopub.execute_input":"2023-10-23T09:11:24.416568Z","iopub.status.idle":"2023-10-23T09:11:24.422452Z","shell.execute_reply.started":"2023-10-23T09:11:24.416546Z","shell.execute_reply":"2023-10-23T09:11:24.421441Z"}}
def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    return torch.device(dev)


# %% [code] {"execution":{"iopub.status.busy":"2023-10-23T09:11:24.423463Z","iopub.execute_input":"2023-10-23T09:11:24.423775Z","iopub.status.idle":"2023-10-23T09:11:24.538804Z","shell.execute_reply.started":"2023-10-23T09:11:24.423751Z","shell.execute_reply":"2023-10-23T09:11:24.537860Z"}}
def train_nn(model, train_loader, test_loader, criterion, optimizer, n_epochs):
    device = set_device()
    best_acc = 0

    for epoch in range(n_epochs):
        print("Epoch number %d" % (epoch + 1))
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            optimizer.zero_grad()

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            running_correct += (labels == predicted).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.00 * running_correct / total
        print(" - Training dataset. Got %d out of %d images correctly (%.3f%%). Epoch loss: %.3f"
              % (running_correct, total, epoch_acc, epoch_loss))

        test_data_acc = evaluate_model_on_test_set(model, test_loader)

        if (test_data_acc > best_acc):
            best_acc = test_data_acc
            save_checkpoint(model, epoch, optimizer, best_acc)

    print("Finished")
    return model


# %% [code] {"execution":{"iopub.status.busy":"2023-10-23T09:11:24.540110Z","iopub.execute_input":"2023-10-23T09:11:24.540416Z","iopub.status.idle":"2023-10-23T09:11:24.552551Z","shell.execute_reply.started":"2023-10-23T09:11:24.540392Z","shell.execute_reply":"2023-10-23T09:11:24.551299Z"}}
def evaluate_model_on_test_set(model, test_loader):
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0
    device = set_device()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted_correctly_on_epoch += (predicted == labels).sum().item()

    epoch_acc = 100.0 * predicted_correctly_on_epoch / total
    print(" - Testing dataset. Got %d out of %d images correctly (%.3f%%)" % (
    predicted_correctly_on_epoch, total, epoch_acc))

    return epoch_acc


# %% [code] {"execution":{"iopub.status.busy":"2023-10-23T09:11:24.553706Z","iopub.execute_input":"2023-10-23T09:11:24.554062Z","iopub.status.idle":"2023-10-23T09:11:24.569606Z","shell.execute_reply.started":"2023-10-23T09:11:24.554028Z","shell.execute_reply":"2023-10-23T09:11:24.568561Z"}}
def save_checkpoint(model, epoch, optimizer, best_acc):
    state = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        # "best_accuracy": best_acc,
        # "optimizer": optimizer.state_dict(),
    }

    torch.save(state, path + '/model_best_checkpoint.pth.tar')


# %% [code] {"execution":{"iopub.status.busy":"2023-10-23T09:11:24.570840Z","iopub.execute_input":"2023-10-23T09:11:24.571205Z","iopub.status.idle":"2023-10-23T09:11:28.123096Z","shell.execute_reply.started":"2023-10-23T09:11:24.571168Z","shell.execute_reply":"2023-10-23T09:11:28.122230Z"}}

resnet18_model = models.resnet18(pretrained=True)
num_ftrs = resnet18_model.fc.in_features
number_of_classes = 5
resnet18_model.fc = nn.Linear(num_ftrs, number_of_classes)
device = set_device()
resnet_18_model = resnet18_model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.003)

# %% [code] {"execution":{"iopub.status.busy":"2023-10-23T10:22:33.774411Z","iopub.execute_input":"2023-10-23T10:22:33.775077Z","iopub.status.idle":"2023-10-23T10:22:33.798294Z","shell.execute_reply.started":"2023-10-23T10:22:33.775043Z","shell.execute_reply":"2023-10-23T10:22:33.797341Z"}}
resnet18_model.to(device)

# %% [code] {"execution":{"iopub.status.busy":"2023-10-23T10:22:38.057166Z","iopub.execute_input":"2023-10-23T10:22:38.057515Z","iopub.status.idle":"2023-10-23T11:04:39.956120Z","shell.execute_reply.started":"2023-10-23T10:22:38.057483Z","shell.execute_reply":"2023-10-23T11:04:39.955125Z"}}
train_nn(resnet18_model, train_loader, test_loader, loss_fn, optimizer, 100)


# %% [code] {"execution":{"iopub.status.busy":"2023-10-23T11:17:35.150705Z","iopub.execute_input":"2023-10-23T11:17:35.151435Z","iopub.status.idle":"2023-10-23T11:17:35.156414Z","shell.execute_reply.started":"2023-10-23T11:17:35.151400Z","shell.execute_reply":"2023-10-23T11:17:35.155441Z"}}
def save_model():
    resnetl8_model = models.resnet18()
    num_ftrs = resnet18_model.fc.in_features
    number_of_classes = 5
    resnet18_model.fc = nn.Linear(num_ftrs, number_of_classes)
    resnet18_model.load_state_dict(check["model"])


# %% [code] {"execution":{"iopub.status.busy":"2023-10-23T11:17:35.759735Z","iopub.execute_input":"2023-10-23T11:17:35.760070Z","iopub.status.idle":"2023-10-23T11:17:36.022689Z","shell.execute_reply.started":"2023-10-23T11:17:35.760045Z","shell.execute_reply":"2023-10-23T11:17:36.021652Z"}}
resnet18_model = models.resnet18()
num_ftrs = resnet18_model.fc.in_features
number_of_classes = 5
resnet18_model.fc = nn.Linear(num_ftrs, number_of_classes)
resnet18_model.load_state_dict(check['model'])
torch.save(resnet18_model, path + '/best_model.pth')

# %% [code] {"execution":{"iopub.status.busy":"2023-10-23T11:17:36.859894Z","iopub.execute_input":"2023-10-23T11:17:36.860758Z","iopub.status.idle":"2023-10-23T11:17:36.928560Z","shell.execute_reply.started":"2023-10-23T11:17:36.860716Z","shell.execute_reply":"2023-10-23T11:17:36.927574Z"}}
# def load_model_checkpoints():
check = torch.load(path + '/model_best_checkpoint.pth.tar')
print(check['epoch'])

# def load_model():
model = torch.load(path + '/best_model.pth')

# %% [code] {"execution":{"iopub.status.busy":"2023-10-23T11:17:54.114505Z","iopub.execute_input":"2023-10-23T11:17:54.115356Z","iopub.status.idle":"2023-10-23T11:17:54.120736Z","shell.execute_reply.started":"2023-10-23T11:17:54.115321Z","shell.execute_reply":"2023-10-23T11:17:54.119725Z"}}
mean = [0.4873, 0.4838, 0.4746]
std = [0.2585, 0.2584, 0.2712]

image_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])


# %% [code] {"execution":{"iopub.status.busy":"2023-10-23T11:17:58.016526Z","iopub.execute_input":"2023-10-23T11:17:58.017485Z","iopub.status.idle":"2023-10-23T11:17:58.024190Z","shell.execute_reply.started":"2023-10-23T11:17:58.017442Z","shell.execute_reply":"2023-10-23T11:17:58.023209Z"}}
def classify(model, image_transforms, image_path, classes):
    model.eval()
    model = model.to(device)
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = model(image.to(device))

    _, predicted = torch.max(output.data, 1)
    #   print(output.data)
    #   print(predicted)
    print(classes[predicted.item()])


# %% [code] {"execution":{"iopub.status.busy":"2023-10-23T12:24:45.336106Z","iopub.execute_input":"2023-10-23T12:24:45.336398Z","iopub.status.idle":"2023-10-23T12:24:45.346174Z","shell.execute_reply.started":"2023-10-23T12:24:45.336373Z","shell.execute_reply":"2023-10-23T12:24:45.345330Z"}}
image_path = '/kaggle/input/old-monk/1.jpg'

# %% [code] {"execution":{"iopub.status.busy":"2023-10-23T11:18:03.430618Z","iopub.execute_input":"2023-10-23T11:18:03.431525Z","iopub.status.idle":"2023-10-23T11:18:03.527605Z","shell.execute_reply.started":"2023-10-23T11:18:03.431488Z","shell.execute_reply":"2023-10-23T11:18:03.526649Z"}}
classify(model, image_transforms, image_path, classes)

# %% [code]
