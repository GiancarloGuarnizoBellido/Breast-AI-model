import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

import utils

def BCEDiceLoss(y_hat, y, weight = 0.1, device = 'cuda'):
    #bce_loss = F.binary_cross_entropy(y_hat, y)
    bce_loss = F.binary_cross_entropy_with_logits(y_hat, y)
    y_hat = torch.sigmoid(y_hat) 
    
    _, dice_loss = utils.dice_coeff_batch(y_hat, y, device)
    loss = bce_loss * weight + dice_loss * (1 - weight)
    return loss
