import torch
import torch.nn as nn
import numpy as np
import math

class inc_avg():
    def __init__(self):
        self.avg = 0.0
        self.weight = 0
        
    def append(self, dat, w = 1):
        self.weight += w
        self.avg = self.avg + (dat - self.avg)*w/self.weight

def init_params(model):
    for p in model.parameters():
        if(p.dim() > 1):
            # nn.init.xavier_normal_(p)
            nn.init.trunc_normal_(p, std = 0.01, a = -0.02, b = 0.02)
        else:
            nn.init.uniform_(p, 0.1, 0.2)

def unif(x, y, device = 'cpu'):
    return 2*torch.rand(x, y, device = device) - 1

def gaus(x, y, device = 'cpu'):
    return torch.normal(0, math.sqrt(2), size = (x,y)).to(device)

def h_sphere(x, y, device = 'cpu'):
    xyz = torch.normal(0, 1, size = (x,y))
    return (xyz/xyz.norm(dim = 1).unsqueeze(1)).to(device)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def make_swiss_roll(num_points,radius_scaling=0.0,num_periods=3,z_max=20.0):
    """
    A quick function to generate swiss roll datasets
    
    Inputs:
        num_points - how many data points to output, integer
        radius_scaling - the effective "radius" of the spiral will be increased proportionally to z multiplied by this 
            constant.  Float
        num_periods - the number of rotations around the origin the spiral will form, float
        z_max - the z values of the data will be uniformly distributed between (0,z_max), float
    Outputs:
        data - a tensor of shape (num_points,3)
    """
    
    t = np.linspace(0,num_periods*2.0*np.pi,num_points)
    x = np.cos(t)*t
    y = np.sin(t)*t
    z = np.random.uniform(low=0.0,high=z_max,size=num_points)
    
    x *= (1.0 + radius_scaling*z)
    y *= (1.0 + radius_scaling*z)
    
    data = np.stack([x,y,z],axis=1)
    
    return data.astype(np.float32)