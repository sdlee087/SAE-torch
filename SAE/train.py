import os, sys, time, logging
sys.path.append('/'.join(os.getcwd().split('/')[:-1]))
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES']='1'

import torch
import torch.nn as nn
import torch.optim as optim

from SAE.SAE import WAE_MMD_abstract
from SAE.dataset import CelebA
from SAE.util import init_params, gaus
from SAE.logging_daily import logging_daily

class WAE_MMD_celebA(WAE_MMD_abstract):
    def __init__(self, network_info, log, device = 'cpu', verbose = 1):
        super(WAE_MMD_celebA, self).__init__(network_info, log, device, verbose)
        self.d = 64
        d = self.d
        self.z_dim = network_info['train']['z_dim']
        self.enc = nn.Sequential(
            nn.Conv2d(3, d, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(d, momentum = 0.9),
            nn.ReLU(True),

            nn.Conv2d(d, 2*d, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(2*d, momentum = 0.9),
            nn.ReLU(True),

            nn.Conv2d(2*d, 4*d, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(4*d, momentum = 0.9),
            nn.ReLU(True),

            nn.Conv2d(4*d, 8*d, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(8*d, momentum = 0.9),
            nn.ReLU(True),
            
            nn.Conv2d(8*d, 16*d, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(16*d, momentum = 0.9),
            nn.ReLU(True),
            
            nn.Flatten(),
            nn.Linear(4*16*d, self.z_dim),
            ).to(device)
        self.dec = nn.Sequential(
            
            nn.Linear(self.z_dim, 16*16*d),
            nn.Unflatten(1, (16*d, 4, 4)),
            
            nn.ConvTranspose2d(16*d, 8*d, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.BatchNorm2d(8*d, momentum = 0.9),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(8*d, 4*d, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.BatchNorm2d(4*d, momentum = 0.9),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(4*d, 2*d, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.BatchNorm2d(2*d, momentum = 0.9),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(2*d, d, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.BatchNorm2d(d, momentum = 0.9),
            nn.ReLU(True),
            
            # reconstruction
            nn.ConvTranspose2d(d, 3, kernel_size = 3, padding = 1),
            nn.Tanh(),
            
            ).to(device)
        init_params(self.enc)
        init_params(self.dec)

        self.hist = network_info['train']['histogram']
        
    # kernel for MMD
    def k(self, x, y, diag = True):
        stat = 0.
        # for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        for scale in [1.]:
            C = scale*2*self.z_dim*2
            kernel = (C/(C + (x.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(dim = 2)))
            if diag:
                stat += kernel.sum()
            else:
                stat += kernel.sum() - kernel.diag().sum()
        return stat

if __name__ == '__main__':
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')

    data_home = '/home/reddragon/data/celebA'
    train_data = CelebA(data_home, 'x_list_train.npy')
    test_data = CelebA(data_home, 'x_list_test.npy')

    # train_loader = torch.utils.data.DataLoader(train_data, batch_size, num_workers = 5, shuffle = True, pin_memory=True, drop_last=True)
    # test_loader  = torch.utils.data.DataLoader(test_data, batch_size, num_workers = 5, shuffle = True, pin_memory=True, drop_last=True)

    logger = logging_daily('./config/log_info.yaml')
    log = logger.get_logging()
    log.setLevel(logging.INFO)

    network_info = {
        'train':{
            'z_sampler':gaus,
            'train_data':train_data,
            'test_data':test_data,
            'batch_size':100
            'z_dim':64,
            
            'encoder_pretrain':False
            'encoder_pretrain_batch_size':1000
            'encoder_pretrain_max_step':200,
            
            'lr':1e-3,
            'beta1':0.5,
            'lambda':100.0,
            
            'lr_schedule':"manual",
            'eps':1.0,
            'L':40,
            'validate':True,
            'histogram':True,
            
            'epoch':55,
            'iter_per_epoch':None,
        },
        'path':{
            'save_best':False,
            'save_path':'./weight.pt',
            'tb_logs':'../tb_logs/wae_celebA-1'
        }
    }

    wae_MMD = WAE_MMD_celebA(network_info, log, device = device)
    wae_MMD.train()