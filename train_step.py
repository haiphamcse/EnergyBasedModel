'''
Train template
'''
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
#TRAIN STEP

def train_step(model,
               dataloader: DataLoader,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
  '''
  Template PyTorch model training for a single epoch.
  Turns a model to training mode and then runs through the
  following steps: forward pass, loss calc, optimizer

  Args:
    model: a PyTorch model
    dataloader:
    loss_fn:
    optimizer:
    device:
  Returns:
    A tuple of training loss and training accuracy
  '''
  model.train()

  #Loop through data loader batches
  for batch in dataloader:
    #Send data to target device
    real_imgs, _ = batch
    real_imgs = real_imgs.to(device)
    small_noise = torch.rand_like(real_imgs) * 0.005
    real_imgs.add_(small_noise).clamp_(min=-1., max=1.)

    #Obtain samples
    fake_imgs = model.sampler.sample_new_exmps(device, steps=60, step_size=10)

    #Predict energy score for all imgs
    inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
    real_out, fake_out = model(inp_imgs).chunk(2, dim=0)
    
    
    #2. Calculate and accumulate loss
    reg_loss = model.alpha * (real_out ** 2 + fake_out ** 2).mean()
    cdiv_loss = fake_out.mean() - real_out.mean()
    loss = reg_loss + cdiv_loss
    
    #3. Optimizer zero grad
    optimizer.zero_grad()

    #4. Loss backward
    loss.backward()

    #5. Optimizer step
    optimizer.step()
    
    #Calculate and accumulate
    return loss
#   Adjust metrics to get avg loss and accuracy per batch
#   train_loss = train_loss / len(dataloader)
#   train_acc = train_acc / len(dataloader)
#   return train_loss, train_acc