import torch
import torch.nn as nn
import torch.distributions as dist
import torchvision.transforms as transforms
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

def unif(x, y, a = -1, b = 1, device = 'cpu'):
    return (b-a)*torch.rand(x, y, device = device) + a

def gaus(x, y, device = 'cpu'):
    return torch.normal(0, math.sqrt(2), size = (x,y)).to(device)

def h_sphere(x, y, device = 'cpu'):
    xyz = torch.normal(0, 1, size = (x,y))
    return (xyz/xyz.norm(dim = 1).unsqueeze(1)).to(device)

def multinomial(x, y, device = 'cpu'):
    return torch.eye(y)[torch.randint(y,(x,))].to(device)    

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
class lap_filter(nn.Module):
    def __init__(self, input_channel = 1, device = 'cpu'):
        super(lap_filter, self).__init__()
        self.filter = nn.Conv2d(1, 1, kernel_size = 3, bias = False)
        self.filter.weight = torch.nn.Parameter(torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).unsqueeze(0).unsqueeze(0))
        self.filter.to(device)
        self.transform = None
        if input_channel == 3:
            self.transform = transforms.functional.rgb_to_grayscale
    
    def forward(self, x):
        if self.transform is not None:
            return self.filter(self.transform(x))
        return self.filter(x)
    
def calculate_sharpness(dataset, batch_size = 256, labeled = False, device = 'cpu', num_workers = 8):
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    if labeled:
        lap = lap_filter(input_channel = dataset[0][0].shape[0], device = device)
        blurr = np.zeros(len(dataset))
        for i, (data, label) in enumerate(dataloader):
            if data.shape[0] == batch_size:
                blurr[(i*batch_size):((i+1)*batch_size)] = lap(data.to(device)).var(axis = (1,2,3)).detach().to('cpu').numpy()
            else: # last batch
                blurr[(i*batch_size):len(dataset)] = lap(data.to(device)).var(axis = (1,2,3)).detach().to('cpu').numpy()
    else:
        lap = lap_filter(input_channel = dataset[0].shape[0], device = device)
        blurr = np.zeros(len(dataset))
        for i, data in enumerate(dataloader):
            if data.shape[0] == batch_size:
                blurr[(i*batch_size):((i+1)*batch_size)] = lap(data.to(device)).var(axis = (1,2,3)).detach().to('cpu').numpy()
            else: # last batch
                blurr[(i*batch_size):len(dataset)] = lap(data.to(device)).var(axis = (1,2,3)).detach().to('cpu').numpy()

    return blurr

def calculate_sharpness_generator(generator, z_sampler, batch_size=50, repeat = 10, device='cpu'):
    blurr = np.empty(batch_size*repeat)
    lap = lap_filter(input_channel = 3, device = device)

    for i in range(repeat):
        batch = generator(z_sampler(batch_size).to(device))
        if batch.shape[1] == 1:
            batch = batch.repeat((1,3,1,1))
        blurr[(i*batch_size):((i+1)*batch_size)] = lap(batch.to(device)).var(axis=(1,2,3)).detach().to('cpu').numpy()
    return blurr

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