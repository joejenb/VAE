from matplotlib.transforms import Transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split

import numpy as np

import wandb
import os

import torchvision
from torchvision import transforms

from VAE import VAE

PATH = "../thumbnails128x128"

wandb.init(project="VAE")
wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release

# WandB – Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config          # Initialize config
config.batch_size = 256          # input batch size for training (default: 64)
config.epochs = 50             # number of epochs to train (default: 10)
config.no_cuda = False         # disables CUDA training
config.seed = 42               # random seed (default: 42)
config.image_size = 128
config.log_interval = 1     # how many batches to wait before logging training status
config.learning_rate = 1e-3
config.momentum = 0.1

config.num_hiddens = 128
config.num_residual_layers = 2
config.num_residual_hiddens = 32
config.num_embeddings = 512
config.num_filters = 64
config.embedding_dim = config.num_filters
config.num_channels = 3
config.data_set = "FFHQ"

def get_data_loaders():
    if config.data_set == "MNIST":
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(config.image_size),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        train_set = torchvision.datasets.MNIST(root="/MNIST/", train=True, download=True, transform=transform)
        val_set = torchvision.datasets.MNIST(root="/MNIST/", train=False, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root="/MNIST/", train=False, download=True, transform=transform)
        num_classes = 10
        config.data_variance = 1

    elif config.data_set == "CIFAR10":
        transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(config.image_size),
                transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
            ])
        train_set = torchvision.datasets.CIFAR10(root="/CIFAR10/", train=True, download=True, transform=transform)
        val_set = torchvision.datasets.CIFAR10(root="/CIFAR10/", train=False, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root="/CIFAR10/", train=False, download=True, transform=transform)
        num_classes = 10
        config.data_variance = np.var(train_set.data / 255.0)

    elif config.data_set == "FFHQ":
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(config.image_size),
                transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
            ])

        dataset = torchvision.datasets.ImageFolder(PATH, transform=transform)
        lengths = [int(len(dataset)*0.7), int(len(dataset)*0.1), int(len(dataset)*0.2)]
        train_set, val_set, test_set = random_split(dataset, lengths)

        config.data_variance = 1#np.var(train_set.data / 255.0)
        num_classes = 0

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, num_classes


def train(model, train_loader, optimiser):

    model.train()
    train_res_recon_error = 0

    for X, _ in train_loader:
        X = X.to(model.device)
        optimiser.zero_grad()

        X_recon, mu, log_var = model(X)

        recon_error = F.mse_loss(X_recon, X) / config.data_variance
        kl_error = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recon_error + 0.001 * kl_error

        loss.backward()
        optimiser.step()
        
        train_res_recon_error += recon_error.item()

    wandb.log({
        "Train Reconstruction Error": train_res_recon_error / len(train_loader.dataset)
    })


def test(model, test_loader):
    # Recall Memory
    model.eval() 

    test_res_recon_error = 0

    example_images = []
    example_reconstructions = []

    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(model.device)

            X_recon, _, _ = model(X)
            recon_error = F.mse_loss(X_recon, X) / config.data_variance
            
            test_res_recon_error += recon_error.item()

        example_images = [wandb.Image(img) for img in X]
        example_reconstructions = [wandb.Image(recon_img) for recon_img in X_recon]

    wandb.log({
        "Test Inputs": example_images,
        "Test Reconstruction": example_reconstructions,
        "Test Reconstruction Error": test_res_recon_error / len(test_loader.dataset)
        })


def main():

    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, val_loader, test_loader, num_classes = get_data_loaders()

    ### Add in correct parameters
    model = VAE(config, device).to(device)
    optimiser = optim.Adam(model.parameters(), lr=config.learning_rate, amsgrad=False)

    wandb.watch(model, log="all")

    for _ in range(config.epochs):
        train(model, train_loader, optimiser)
        test(model, test_loader)

if __name__ == '__main__':
    main()