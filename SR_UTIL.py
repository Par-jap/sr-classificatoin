from torch import nn, optim
from torch.nn.functional import relu
import matplotlib.pyplot as plt

from torch.autograd import Variable
from math import log10
import torch
import os
import time

import numpy as np
import wandb


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, padding=2)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, val=0)

    def forward(self, x):
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = relu(x)
        x = self.conv3(x)
        x = relu(x)
        return x

def sec2min(sec):
  min = sec//60
  sec %= 60
  return "{}m {}s".format(int(min), int(sec))


def sr_train(model,  train_loader, val_loader, optimizer, epoch=1000, loss=nn.MSELoss(), wandb_config=True, step=100):
  if(wandb_config):
    w_log = wandb.login()
  else:
    w_log = False

  _step = step-1
  if(w_log):
    if wandb_config:
        wandb.init(project="house_server")
    
  total_start = time.time()

  """
  arguments:  model,  train_loader, val_loader, optimizer, epoch=1000, loss=nn.MSELoss(), wandb_config=None
  returns:    losses, psnrs
  """

  criterion = nn.MSELoss()
  psnrs = []
  losses = []

  for e in range(epoch):
    if(e%step==0):
      time_step_start = time.time()
      e10_loss, e10_psnr = 0, 0

    model.train()
    epoch_loss, epoch_psnr = 0, 0
    for batch in train_loader:
        inputs, targets = Variable(batch[0]), Variable(batch[1])
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()        
        prediction = model(inputs)
        loss = criterion(prediction, targets)
        
        epoch_loss += loss.data
        epoch_psnr += 10 * log10(1 / loss.data)

        loss.backward()
        optimizer.step()
    if(w_log):
        l = epoch_loss / len(train_loader)
        p = epoch_psnr / len(train_loader)
        wandb.log({'loss': l, 'psnr': p})
    e10_loss += epoch_loss
    e10_psnr += epoch_psnr
    psnrs.append(epoch_psnr / len(train_loader))
    losses.append((epoch_loss/ len(train_loader)).to("cpu").detach().numpy())

    if(e%step==_step):
      model.eval()
      val_loss, val_psnr = 0, 0
      with torch.no_grad():
          for batch in val_loader:
            
            inputs, targets = batch[0], batch[1]    
            inputs = inputs.to(device)
            targets = targets.to(device) 
            
            prediction = model(inputs)
            loss = criterion(prediction, targets)
            val_loss += loss.data
            val_psnr += 10 * log10(1 / loss.data)

      time_step_end = time.time()
      t = (time_step_end-time_step_start)/step
      last = (time_step_end-total_start)*(epoch/(e+1) - 1)
      print('[Epoch {}] Loss: {:.4f}, PSNR: {:.4f} dB'.format(e + 1, 0.1*e10_loss / len(train_loader), 0.1*e10_psnr / len(train_loader)))
      print("===> Avg. Loss: {:.4f}, PSNR: {:.4f} dB".format(val_loss / len(val_loader), val_psnr / len(val_loader)))
      print("===> time/epoc: {:.2f}[s], left time: {}".format(t, sec2min(last)))
        
  total_end = time.time()
  total_time = total_end - total_start
  return losses, psnrs, total_time