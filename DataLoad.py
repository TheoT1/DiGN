# DataLoader construction for various datasets.
#
# Last updated: Nov 5 2021
import torch
import torch.nn as nn
from torchvision import datasets, transforms

TRAIN_TRANSFORMS_DEFAULT = lambda size: transforms.Compose([
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25,.25,.25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ])

TEST_TRANSFORMS_DEFAULT = lambda size:transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ])
    
# Special transforms for ImageNet(s)
TRAIN_TRANSFORMS_IMAGENET = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1
        ),
        transforms.ToTensor(),
    ])

TEST_TRANSFORMS_IMAGENET = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
def get_loader_from_path(path, batch_size=128, num_workers=4, d_input=64):
    """ Returns data loader for given path """
    dataset = datasets.ImageFolder(path, transform=TEST_TRANSFORMS_DEFAULT(d_input))
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def get_loaders_imagenette(data_path, batch_size_train=128, batch_size_val=128, num_workers=4):
    """ Returns data loaders for train and val sets of ImageNette"""
    train_dataset = datasets.ImageFolder(data_path+'/train', transform=TRAIN_TRANSFORMS_IMAGENET)
    val_dataset   = datasets.ImageFolder(data_path+'/val', transform=TEST_TRANSFORMS_IMAGENET)
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, num_workers=num_workers, shuffle=True, drop_last=True)
    val_loader    = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_val, num_workers=num_workers, shuffle=False)
    return train_loader, val_loader


def get_loaders_tinyimagenet(data_path, batch_size_train=128, batch_size_val=128, num_workers=4):
    """ Returns data loaders for train and val sets of Tiny-ImageNet"""
    train_dataset = datasets.ImageFolder(data_path+'/train', transform=TRAIN_TRANSFORMS_DEFAULT(64))
    val_dataset   = datasets.ImageFolder(data_path+'/val', transform=TEST_TRANSFORMS_DEFAULT(64))
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, num_workers=num_workers, shuffle=True, drop_last=True)
    val_loader    = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_val, num_workers=num_workers, shuffle=False)
    return train_loader, val_loader


def get_loaders_cifar10(data_path, batch_size_train=128, batch_size_val=128, num_workers=4):
    """ Returns data loaders for train and val sets of CIFAR-10"""
    train_dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=TRAIN_TRANSFORMS_DEFAULT(32))
    val_dataset   = datasets.CIFAR10(data_path, train=False, download=True, transform=TEST_TRANSFORMS_DEFAULT(32))
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, num_workers=num_workers, shuffle=True, drop_last=True)
    val_loader    = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_val, num_workers=num_workers, shuffle=False)
    return train_loader, val_loader


def get_loaders_cifar100(data_path, batch_size_train=128, batch_size_val=128, num_workers=4):
    """ Returns data loaders for train and val sets of CIFAR-100"""
    train_dataset = datasets.CIFAR100(data_path, train=True, download=True, transform=TRAIN_TRANSFORMS_DEFAULT(32))
    val_dataset   = datasets.CIFAR100(data_path, train=False, download=True, transform=TEST_TRANSFORMS_DEFAULT(32))
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, num_workers=num_workers, shuffle=True, drop_last=True)
    val_loader    = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_val, num_workers=num_workers, shuffle=False)
    return train_loader, val_loader
    

def get_loader_from_numpy(x,y,batch_size=128, num_workers=4):
    """ Returns data loaders using numpy data as input - for corrupted image data test """
    tensor_x      = torch.Tensor(x)/255.0 # transform to torch tensor
    tensor_y      = torch.Tensor(y).type(torch.int64)
    my_dataset    = torch.utils.data.TensorDataset(tensor_x,tensor_y) # create your dataset
    my_dataloader = torch.utils.data.DataLoader(my_dataset,batch_size=batch_size, num_workers=num_workers, shuffle=False) # create your dataloader
    return my_dataloader
