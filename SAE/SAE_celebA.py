import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

from .SAE import SAE_abstract
from .SAE import WAE_MMD_abstract
from .dataset import CelebA
from .util import init_params, inc_avg

import numpy as np

class SAE_celebA(SAE_abstract):
    def __init__(self, network_info, log, device = 'cpu', verbose = 1):
        super(SAE_celebA, self).__init__(network_info, log, device, verbose)
        self.d = 64
        d = self.d
        self.z_dim = network_info['train']['z_dim']
        self.enc = nn.Sequential(
            nn.Conv2d(3, d, kernel_size = 5, stride = 2, padding = 2),
            nn.BatchNorm2d(d),
            nn.ReLU(True),

            nn.Conv2d(d, 2*d, kernel_size = 5, stride = 2, padding = 2),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True),

            nn.Conv2d(2*d, 4*d, kernel_size = 5, stride = 2, padding = 2),
            nn.BatchNorm2d(4*d),
            nn.ReLU(True),

            nn.Conv2d(4*d, 8*d, kernel_size = 5, stride = 2, padding = 2),
            nn.BatchNorm2d(8*d),
            nn.ReLU(True),
            
            nn.Conv2d(8*d, 4*d, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(4*d),
            nn.ReLU(True),
            
            nn.Flatten(),
            nn.Linear(16*4*d, self.z_dim),
            ).to(device)
        self.dec = nn.Sequential(
            
            nn.Linear(self.z_dim, 64*8*d),
            nn.Unflatten(1, (8*d, 8, 8)),
            
            nn.ConvTranspose2d(8*d, 4*d, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),
            nn.BatchNorm2d(4*d),
            nn.LeakyReLU(0.1, True),
            
            nn.ConvTranspose2d(4*d, 2*d, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),
            nn.BatchNorm2d(2*d),
            nn.LeakyReLU(0.1, True),
            
            nn.Conv2d(2*d, 2*d, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(2*d),
            nn.LeakyReLU(0.1, True),
            
            nn.ConvTranspose2d(2*d, d, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.1, True),
            
            nn.Conv2d(d, d, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.1, True),
            
            # reconstruction
            nn.ConvTranspose2d(d, 3, kernel_size = 3, padding = 1),
            nn.Sigmoid(),
            
            ).to(device)
        init_params(self.enc)
        init_params(self.dec)

        self.hist = network_info['train']['histogram']


class SAE_celebA_v0(SAE_celebA):
    def __init__(self, network_info, log, device = 'cpu', verbose = 1):
        super(SAE_celebA_v0, self).__init__(network_info, log, device, verbose)
        self.d = 64
        d = self.d

        self.enc = nn.Sequential(
            nn.Conv2d(3, d, kernel_size = 5, stride = 2, padding = 2),
            nn.BatchNorm2d(d),
            nn.ReLU(True),

            nn.Conv2d(d, 2*d, kernel_size = 5, stride = 2, padding = 2),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True),

            nn.Conv2d(2*d, 4*d, kernel_size = 5, stride = 2, padding = 2),
            nn.BatchNorm2d(4*d),
            nn.ReLU(True),

            nn.Conv2d(4*d, 8*d, kernel_size = 5, stride = 2, padding = 2),
            nn.BatchNorm2d(8*d),
            nn.ReLU(True),
            
            nn.Conv2d(8*d, 4*d, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(4*d),
            nn.ReLU(True),
            
            nn.Flatten(),
            nn.Linear(16*4*d, self.z_dim),
            ).to(device)
        self.dec = nn.Sequential(
            
            nn.Linear(self.z_dim, 64*8*d),
            nn.Unflatten(1, (8*d, 8, 8)),
            
            nn.ConvTranspose2d(8*d, 4*d, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),
            nn.BatchNorm2d(4*d),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(4*d, 2*d, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True),
            
            nn.Conv2d(2*d, 2*d, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(2*d, d, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),
            nn.BatchNorm2d(d),
            nn.ReLU(True),
            
            nn.Conv2d(d, d, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(d),
            nn.ReLU(True),
            
            # reconstruction
            nn.ConvTranspose2d(d, 3, kernel_size = 3, padding = 1),
            nn.Sigmoid(),
            
            ).to(device)
        init_params(self.enc)
        init_params(self.dec)

class WAE_MMD_celebA(WAE_MMD_abstract):
    def __init__(self, network_info, log, device = 'cpu', verbose = 1):
        super(WAE_MMD_celebA, self).__init__(network_info, log, device, verbose)
        self.d = 64
        d = self.d
        self.z_dim = network_info['train']['z_dim']
        self.enc = nn.Sequential(
            nn.Conv2d(3, d, kernel_size = 5, stride = 2, padding = 2),
            nn.BatchNorm2d(d),
            nn.ReLU(True),

            nn.Conv2d(d, 2*d, kernel_size = 5, stride = 2, padding = 2),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True),

            nn.Conv2d(2*d, 4*d, kernel_size = 5, stride = 2, padding = 2),
            nn.BatchNorm2d(4*d),
            nn.ReLU(True),

            nn.Conv2d(4*d, 8*d, kernel_size = 5, stride = 2, padding = 2),
            nn.BatchNorm2d(8*d),
            nn.ReLU(True),
            
            nn.Conv2d(8*d, 4*d, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(4*d),
            nn.ReLU(True),
            
            nn.Flatten(),
            nn.Linear(16*4*d, self.z_dim),
            ).to(device)
        self.dec = nn.Sequential(
            
            nn.Linear(self.z_dim, 64*8*d),
            nn.Unflatten(1, (8*d, 8, 8)),
            
            nn.ConvTranspose2d(8*d, 4*d, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),
            nn.BatchNorm2d(4*d),
            nn.LeakyReLU(0.1, True),
            
            nn.ConvTranspose2d(4*d, 2*d, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),
            nn.BatchNorm2d(2*d),
            nn.LeakyReLU(0.1, True),
            
            nn.Conv2d(2*d, 2*d, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(2*d),
            nn.LeakyReLU(0.1, True),
            
            nn.ConvTranspose2d(2*d, d, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.1, True),
            
            nn.Conv2d(d, d, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.1, True),
            
            # reconstruction
            nn.ConvTranspose2d(d, 3, kernel_size = 3, padding = 1),
            nn.Sigmoid(),
            
            ).to(device)
        init_params(self.enc)
        init_params(self.dec)

        self.hist = network_info['train']['histogram']
        
    def k(self, x, y):
        C = 2*self.z_dim*2
        return (C/(C + (x.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(dim = 2))).sum()


class WAE_MMD_celebA_v0(WAE_MMD_abstract):
    def __init__(self, network_info, log, device = 'cpu', verbose = 1):
        super(WAE_MMD_celebA_v0, self).__init__(network_info, log, device, verbose)
        self.d = 64
        d = self.d
        self.z_dim = network_info['train']['z_dim']
        self.enc = nn.Sequential(
            nn.Conv2d(3, d, kernel_size = 5, stride = 2, padding = 2),
            nn.BatchNorm2d(d),
            nn.ReLU(True),

            nn.Conv2d(d, 2*d, kernel_size = 5, stride = 2, padding = 2),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True),

            nn.Conv2d(2*d, 4*d, kernel_size = 5, stride = 2, padding = 2),
            nn.BatchNorm2d(4*d),
            nn.ReLU(True),

            nn.Conv2d(4*d, 8*d, kernel_size = 5, stride = 2, padding = 2),
            nn.BatchNorm2d(8*d),
            nn.ReLU(True),
            
            nn.Conv2d(8*d, 4*d, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(4*d),
            nn.ReLU(True),
            
            nn.Flatten(),
            nn.Linear(16*4*d, self.z_dim),
            ).to(device)
        self.dec = nn.Sequential(
            
            nn.Linear(self.z_dim, 64*8*d),
            nn.Unflatten(1, (8*d, 8, 8)),
            
            nn.ConvTranspose2d(8*d, 4*d, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),
            nn.BatchNorm2d(4*d),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(4*d, 2*d, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True),
            
            nn.Conv2d(2*d, 2*d, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(2*d, d, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),
            nn.BatchNorm2d(d),
            nn.ReLU(True),
            
            nn.Conv2d(d, d, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(d),
            nn.ReLU(True),
            
            # reconstruction
            nn.ConvTranspose2d(d, 3, kernel_size = 3, padding = 1),
            nn.Sigmoid(),
            
            ).to(device)
        init_params(self.enc)
        init_params(self.dec)

        self.hist = network_info['train']['histogram']
        
    def k(self, x, y):
        C = 2*self.z_dim*2
        return (C/(C + (x.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(dim = 2))).sum()

if __name__ == '__main__':
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')

    data_home = '/home/reddragon/data/celebA'
    all_data = CelebA(data_home)
    train_n = int(len(all_data)*0.7)
    train_data, test_data = torch.utils.data.random_split(all_data, [train_n, len(all_data) - train_n])
    batch_size = 300

    rand_sampler1 = torch.utils.data.RandomSampler(train_data, num_samples = batch_size, replacement = True)
    rand_sampler2 = torch.utils.data.RandomSampler(test_data, num_samples = batch_size, replacement = True)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size, num_workers = 10, sampler = rand_sampler1)
    test_loader  = torch.utils.data.DataLoader(test_data, batch_size, num_workers = 10, sampler = rand_sampler2)

    network_info = {
        'train':{
            'z_sampler':gaus,
            'train_generator':train_loader,
            'test_generator':test_loader,
            
            'lr':0.001,
            'beta1':0.9,
            'lambda':1e-3,
            'lambda_exp':None,
            'eps':1e-3,
            'L':400,
            'validate':True,

            'histogram':True,
            
            'epoch':100,
            'iter_per_epoch':50,
        },
        'path':{
            'save_best':True,
            'save_path':'./sae_celebA-v1-test.pt',
            'tb_logs':'./tb_logs/sae_swiss'
        }
    }

    sae_celebA_v1 = SAE_celebA(network_info, device = device)
    start = time.time()
    sae_celebA_v1.train()
    print("Elapsed time: %.3fs" % (time.time() - start))

