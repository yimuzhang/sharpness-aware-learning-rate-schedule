import numpy as np
from torchvision.datasets import CIFAR10,MNIST,CocoCaptions
from typing import Tuple
from torch.utils.data.dataset import TensorDataset
import os
import torch
from torch import Tensor
import torch.nn.functional as F
import pycocotools
from coloredmnist import ColoredDataset_generated
from torchvision import transforms

#DATASETS_FOLDER = os.environ["DATASETS"]
DATASETS_FOLDER ="E:/PEK_project/dataset"



def center(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(0)
    return X_train - mean, X_test - mean

def standardize(X_train: np.ndarray, X_test: np.ndarray):
    std = X_train.std(0)
    tmp=(std==0)
    std=std+tmp
   # print(X_train[0],tmp.std(0))
   # print("syd:",std)
    #print(std)
    return (X_train / std, X_test / std)

def flatten(arr: np.ndarray):
    return arr.reshape(arr.shape[0], -1)

def unflatten(arr: np.ndarray, shape: Tuple):
    return arr.reshape(arr.shape[0], *shape)

def _one_hot(tensor: Tensor, num_classes: int, default=0):
    M = F.one_hot(tensor, num_classes)
    M[M == 0] = default
    return M.float()

def make_labels(y, loss):
    if loss == "ce":
        return y
    elif loss == "mse":
        return _one_hot(y, 10, 0)


def load_cifar(loss: str) -> (TensorDataset, TensorDataset):
    cifar10_train = CIFAR10(root=DATASETS_FOLDER, download=True, train=True)
    cifar10_test = CIFAR10(root=DATASETS_FOLDER, download=True, train=False)
    X_train, X_test = flatten(cifar10_train.data / 255), flatten(cifar10_test.data / 255)
    y_train, y_test = make_labels(torch.tensor(cifar10_train.targets), loss), \
        make_labels(torch.tensor(cifar10_test.targets), loss)
    center_X_train, center_X_test = center(X_train, X_test)
    standardized_X_train, standardized_X_test = standardize(center_X_train, center_X_test)
    train = TensorDataset(torch.from_numpy(unflatten(standardized_X_train, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_train)
    test = TensorDataset(torch.from_numpy(unflatten(standardized_X_test, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_test)
    return train, test

def load_mnist(loss: str) -> (TensorDataset, TensorDataset):
    cifar10_train = MNIST(root="E:/PEK_project/dataset", download=True, train=True)
    cifar10_test = MNIST(root="E:/PEK_project/dataset", download=True, train=False)
    X_train, X_test = flatten(cifar10_train.data / 1), flatten(cifar10_test.data / 1)
   # tmp=torch.zeros(60000,240)
  #  X_train=torch.cat((X_train,tmp),dim=1)
   # X_train=X_train.repeat(1,3)
   # tmp=torch.zeros(10000,240)
    #X_test=torch.cat((X_test,tmp),dim=1)
   # X_test=X_test.repeat(1,3)

    #print("123",X_train.std(0))
    y_train, y_test = make_labels(torch.tensor(cifar10_train.targets), loss), \
        make_labels(torch.tensor(cifar10_test.targets), loss)
    center_X_train, center_X_test = center(X_train, X_test)
    #print(center_X_train)
    standardized_X_train, standardized_X_test = standardize(center_X_train, center_X_test)
    train = TensorDataset((unflatten(standardized_X_train, (28,28))).float(), y_train)
    test = TensorDataset((unflatten(standardized_X_test, (28,28))).float(), y_test)
    return train, test

def load_coco(loss: str) -> (TensorDataset, TensorDataset):
    cifar10_train = CocoCaptions(root="E:\PEK_project\dataset\\annotations\stuff_train2017_pixelmaps\stuff_train2017_pixelmaps",annFile="E:\PEK_project\dataset\\annotations\stuff_train2017.json")
    cifar10_test = CocoCaptions(root="E:\PEK_project\dataset\\annotations\stuff_train2017_pixelmaps\stuff_train2017_pixelmaps",annFile="E:\PEK_project\dataset\\annotations\stuff_train2017.json")
    X_train, X_test = flatten(cifar10_train.data / 255), flatten(cifar10_test.data / 255)
    y_train, y_test = make_labels(torch.tensor(cifar10_train.targets), loss), \
        make_labels(torch.tensor(cifar10_test.targets), loss)
    center_X_train, center_X_test = center(X_train, X_test)
    standardized_X_train, standardized_X_test = standardize(center_X_train, center_X_test)
    train = TensorDataset((unflatten(standardized_X_train, (28,28))).float(), y_train)
    test = TensorDataset((unflatten(standardized_X_test, (28,28))).float(), y_test)
    return train, test

def load_color(loss: str) -> (TensorDataset, TensorDataset):
    cifar10_train = MNIST(root="E:/PEK_project/dataset", download=True, train=True)
    cifar10_test = MNIST(root="E:/PEK_project/dataset", download=True, train=False)
    color_train=ColoredDataset_generated(cifar10_train,var=0.2)
    color_test=ColoredDataset_generated(cifar10_test,var=0.2)
    train_loader = torch.utils.data.DataLoader(color_train,
                                              batch_size=5000, shuffle=True,
                                              num_workers=0)
    test_loader = torch.utils.data.DataLoader(color_test,
                                              batch_size=1000, shuffle=True,
                                              num_workers=0)
    train_data=next(iter(train_loader))
    test_data=next(iter(test_loader))
    X_train, X_test = flatten(train_data / 1), flatten(test_data / 1)
    #print("123",X_train.std(0))
    y_train, y_test = make_labels(torch.tensor(color_train.targets), loss), \
        make_labels(torch.tensor(color_test.targets), loss)
    center_X_train, center_X_test = center(X_train, X_test)
    #print(center_X_train)
    standardized_X_train, standardized_X_test = standardize(center_X_train, center_X_test)
    train = TensorDataset((unflatten(standardized_X_train, (28,28))).float(), y_train)
    test = TensorDataset((unflatten(standardized_X_test, (28,28))).float(), y_test)
    return train, test