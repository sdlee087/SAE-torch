import os, sys, time, logging
sys.path.append('/'.join(os.getcwd().split('/')[:-2]))
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES']='1'

import torch
import torch.nn as nn
import torch.optim as optim

from SAE.SAE import CWAE_MMD_abstract
from SAE.dataset import labeled_MNIST
from SAE.util import init_params, gaus, multinomial
from SAE.logging_daily import logging_daily

class CWAE_MMD_MNIST(CWAE_MMD_abstract):
    def __init__(self, network_info, log, device = 'cpu', verbose = 1):
        super(CWAE_MMD_MNIST, self).__init__(network_info, log, device, verbose)
        self.d = 64
        d = self.d
        self.embed_data = nn.Identity().to(device)
        self.embed_label = nn.Sequential(
            nn.Linear(10, 28*28),
            nn.Unflatten(1, (1,28,28)),
        ).to(device)
        self.enc = nn.Sequential(
            nn.Conv2d(2, d, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(d),
            nn.ReLU(True),

            nn.Conv2d(d, 2*d, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True),

            nn.Conv2d(2*d, 4*d, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(4*d),
            nn.ReLU(True),

            nn.Conv2d(4*d, 8*d, kernel_size = 3, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(8*d),
            nn.ReLU(True),
            
            nn.Flatten(),
            nn.Linear(4*8*d, self.z_dim)
            ).to(device)
                                
        self.dec = nn.Sequential(
            nn.Linear(self.z_dim + self.y_dim, 49*8*d),
            # nn.Linear(self.z_dim + self.y_dim, 1024, bias = False),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(True),
            
            # nn.Linear(1024, 49*8*d),
            nn.Unflatten(1, (8*d, 7, 7)),
            
            nn.ConvTranspose2d(8*d, 4*d, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(4*d),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(4*d, 2*d, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True),
            
            # reconstruction
            nn.Conv2d(2*d, 1, kernel_size = 3, padding = 1),
            nn.Tanh(),
            
            ).to(device)
        # init_params(self.embed_data)
        init_params(self.embed_label)
        init_params(self.enc)
        init_params(self.dec)

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

if __name__ == '__main__':
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')

    data_home = '/home/reddragon/data/MNIST'
    train_data = labeled_MNIST(data_home, train = True)
    test_data = labeled_MNIST(data_home, train = False)

    logger = logging_daily('./config/log_info.yaml')
    log = logger.get_logging()
    log.setLevel(logging.INFO)

    network_info = {
        'train':{
            'z_sampler':gaus,
            'y_sampler':multinomial,
            'train_data':train_data,
            'test_data':test_data,
            'batch_size':100,
            'z_dim':6,
            'y_dim':10,
            
            'encoder_pretrain':False,
            'encoder_pretrain_batch_size':1000,
            'encoder_pretrain_max_step':200,
            
            'lr':5e-4,
            'beta1':0.9,
            'lambda':10.0,
            
            'lr_schedule':"manual",
            'validate':True,
            'histogram':True,
            
            'epoch':10,
            'iter_per_epoch':None,
        },
        'path':{
            'save_best':False,
            'save_path':'./weight.pt',
            'tb_logs':'../tb_logs/train_base'
        }
    }

    cwae_MMD = CWAE_MMD_MNIST(network_info, log, device = device)
    cwae_MMD.train()