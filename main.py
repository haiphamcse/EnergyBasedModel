import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
import os
import json
import math
import numpy as np
import random

from model import *
from train import *

def createdataset(BATCH_SIZE: int = 32):
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
  ])
  train_set = MNIST("/",train=True, transform=transform, download=True)
  test_set = MNIST("/",train=False, transform=transform, download=True)
  
  train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

  return train_loader, test_loader

if __name__ == '__main__':
    print("TRAINING")
    train_loader, test_loader = createdataset()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepEnergyModel(img_shape=(1,28,28), batch_size=32).to(device)
    #print(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0., 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97) # Exponential decay over epochs
    train(model=model,
          train_dataloader=train_loader,
          optimizer=optimizer,
          epochs=60,
          device=device)
    