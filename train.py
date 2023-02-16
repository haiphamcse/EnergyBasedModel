
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from train_step import *
import random
import torchvision

def train(model,
          train_dataloader,
          optimizer,
          epochs,
          device,
          writer):
    '''
    Train and test PyTorch model
    Pass a model through train_step() and test_step()
    Args:
    Returns:
    '''
    #Loop through training and training steps
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model=model,
                                dataloader=train_dataloader,
                                optimizer=optimizer,
                                device=device)
        #Debug 
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
        )
        
        #Generate callback
        
        #Sampler callback
        
        #Outlier callback
    
