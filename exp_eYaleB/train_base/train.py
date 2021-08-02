import os, sys, time, logging
sys.path.append('/'.join(os.getcwd().split('/')[:-2]))
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES']='1'

import torch
import torch.nn as nn
import torch.optim as optim

from SAE.SAE import CWAE_MMD_abstract
from SAE.dataset import eYaleB128
from SAE.util import init_params, gaus, unif, multinomial
from SAE.logging_daily import logging_daily

class CWAE_MMD_eYaleB(CWAE_MMD_abstract):
    def __init__(self, network_info, log, device = 'cpu', verbose = 1):
        super(CWAE_MMD_eYaleB, self).__init__(network_info, log, device, verbose)
        self.d = 64
        d = self.d
        self.embed_data = nn.Sequential(
            nn.Conv2d(1, d, kernel_size = 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(d),
            nn.ReLU(True),
        ).to(device)
        self.embed_label = nn.Sequential(
            nn.Linear(self.y_dim, 64*64),
            nn.Unflatten(1, (1,64,64)),
        ).to(device)
        self.enc = nn.Sequential(

            nn.Conv2d(d+1, 2*d, kernel_size = 5, stride = 2, padding = 2, bias = False),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True),

            nn.Conv2d(2*d, 4*d, kernel_size = 5, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(4*d),
            nn.ReLU(True),
            
            nn.Conv2d(4*d, 4*d, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(4*d),
            nn.ReLU(True),

            nn.Conv2d(4*d, 8*d, kernel_size = 3, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(8*d),
            nn.ReLU(True),
            
            nn.Conv2d(8*d, 16*d, kernel_size = 3, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(16*d),
            nn.ReLU(True),
            
            nn.Flatten(),
            nn.Linear(16*16*d, self.z_dim)
            ).to(device)
                                
        self.dec = nn.Sequential(
            nn.Linear(self.z_dim + self.y_dim, 64*16*d),
            nn.Unflatten(1, (16*d, 8, 8)),
            
            nn.ConvTranspose2d(16*d, 8*d, kernel_size = 3, stride = 2, padding = 1, output_padding = 1, bias = False),
            nn.BatchNorm2d(8*d),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(8*d, 4*d, kernel_size = 3, stride = 2, padding = 1, output_padding = 1, bias = False),
            nn.BatchNorm2d(4*d),
            nn.ReLU(True),
            
            nn.Conv2d(4*d, 4*d, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(4*d),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(4*d, 2*d, kernel_size = 5, stride = 2, padding = 2, output_padding = 1, bias = False),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(2*d, d, kernel_size = 5, stride = 2, padding = 2, output_padding = 1, bias = False),
            nn.BatchNorm2d(d),
            nn.ReLU(True),
            
            nn.Conv2d(d, d, kernel_size = 5, padding = 2, bias = False),
            nn.BatchNorm2d(d),
            nn.ReLU(True),
            
            # reconstruction
            nn.Conv2d(d, 1, kernel_size = 7, padding = 3),
            nn.Tanh(),
            
            ).to(device)

        self.hist = network_info['train']['histogram']
        
    # kernel for MMD
    def k(self, x, y, diag = True):
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = scale*2*self.z_dim*2
            kernel = (C/(C + (x.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(dim = 2)))
            if diag:
                stat += kernel.sum()
            else:
                stat += kernel.sum() - kernel.diag().sum()
        return stat

def multi_and_unif(x, y, device = 'cpu'):
    return torch.cat((multinomial(x, y-2, device), unif(x, 1, -130/180, 130/180, device), unif(x, 1, -40/90, 1, device)), axis = 1)

if __name__ == '__main__':
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')

    data_home = '/home/reddragon/data/YaleBFace128'
    train_data = eYaleB128(data_home, train = True)
    test_data = eYaleB128(data_home, train = False)

    logger = logging_daily('./config/log_info.yaml')
    log = logger.get_logging()
    log.setLevel(logging.INFO)

    network_info = {
        'train':{
            'z_sampler':gaus,
            'y_sampler':multi_and_unif,
            'train_data':train_data,
            'test_data':test_data,
            'batch_size':64,
            'z_dim':16,
            'y_dim':40,
            
            'encoder_pretrain':False,
            'encoder_pretrain_batch_size':1000,
            'encoder_pretrain_max_step':200,
            
            'lr':1e-3,
            'beta1':0.9,
            'lambda':0.0,
            
            'lr_schedule':"manual",
            'validate':True,
            'histogram':True,
            
            'epoch':100,
            'iter_per_epoch':None,
        },
        'path':{
            'save_best':False,
            'save_path':'./weight.pt',
            'tb_logs':'../tb_logs/train_base'
        }
    }

    cwae_MMD = CWAE_MMD_eYaleB(network_info, log, device = device)
    cwae_MMD.train()