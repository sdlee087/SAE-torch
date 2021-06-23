import os, sys, time
sys.path.append('/'.join(os.getcwd().split('/')))
os.environ['CUDA_VISIBLE_DEVICES']='2'

import torch
import torch.nn as nn
import torch.optim as optim
from SAE import *
from util import unif
from util import make_swiss_roll

# def unif(x, y, device = 'cpu'):
#     return 2*torch.rand(x, y, device = device) - 1

# def make_swiss_roll(num_points,radius_scaling=0.0,num_periods=3,z_max=20.0):
#     """
#     A quick function to generate swiss roll datasets
    
#     Inputs:
#         num_points - how many data points to output, integer
#         radius_scaling - the effective "radius" of the spiral will be increased proportionally to z multiplied by this 
#             constant.  Float
#         num_periods - the number of rotations around the origin the spiral will form, float
#         z_max - the z values of the data will be uniformly distributed between (0,z_max), float
#     Outputs:
#         data - a tensor of shape (num_points,3)
#     """
    
#     t = np.linspace(0,num_periods*2.0*np.pi,num_points)
#     x = np.cos(t)*t
#     y = np.sin(t)*t
#     z = np.random.uniform(low=0.0,high=z_max,size=num_points)
    
#     x *= (1.0 + radius_scaling*z)
#     y *= (1.0 + radius_scaling*z)
    
#     data = np.stack([x,y,z],axis=1)
    
#     return data.astype(np.float32)

class WAE_MMD_swiss(WAE_MMD_abstract):
    def __init__(self, network_info, device = 'cpu'):
        super(WAE_MMD_swiss, self).__init__(network_info, device)
    
        self.enc = nn.Sequential(
            nn.Linear(3, 50),
            nn.ReLU(True),
            nn.Linear(50, 50),
            nn.ReLU(True),
            nn.Linear(50, 2),
        ).to(device)

        self.dec = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(True),
            nn.Linear(50, 50),
            nn.ReLU(True),
            nn.Linear(50, 3),
        ).to(device)
        
        init_params(self.enc)
        init_params(self.dec)
        
        self.z_dim = 2
        
    def k(self, x, y):
        C = 2
        return (C/(C + (x.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(dim = 2))).sum()

class SAE_swiss(SAE_abstract):
    def __init__(self, network_info, device = 'cpu'):
        super(SAE_swiss, self).__init__(network_info, device)

        self.enc = nn.Sequential(
                nn.Linear(3, 50),
                nn.ReLU(True),
                nn.Linear(50, 50),
                nn.ReLU(True),
                nn.Linear(50, 2),
            ).to(device)

        self.dec = nn.Sequential(
            nn.Linear(2, 50),
#             nn.ReLU(True),
            nn.LeakyReLU(0.1, True),
            nn.Linear(50, 50),
#             nn.ReLU(True),
            nn.LeakyReLU(0.1, True),
            nn.Linear(50, 3),
        ).to(device)

        init_params(self.enc)
        init_params(self.dec)

        self.z_dim = 2

class SAE_swiss_v1(SAE_abstract):
    def __init__(self, network_info, device = 'cpu'):
        super(SAE_swiss_v1, self).__init__(network_info, device)

        self.enc = nn.Sequential(
                nn.Linear(3, 50),
                nn.ReLU(True),
                nn.Linear(50, 50),
                nn.ReLU(True),
                nn.Linear(50, 2),
            ).to(device)

        self.dec = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(True),
#             nn.LeakyReLU(0.1, True),
            nn.Linear(50, 50),
            nn.ReLU(True),
#             nn.LeakyReLU(0.1, True),
            nn.Linear(50, 3),
        ).to(device)

        init_params(self.enc)
        init_params(self.dec)

        self.z_dim = 2

if __name__ == '__main__':
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')

    num_points_train = 16384
    data_train = make_swiss_roll(num_points_train,num_periods=2)
    num_points_test = 1024
    data_test = make_swiss_roll(num_points_test,num_periods=2)

    swiss_roll_train = ToyDataset(data_train)
    swiss_roll_test = ToyDataset(data_test)

    batch_size = 512

    rand_sampler = torch.utils.data.RandomSampler(swiss_roll_train, num_samples=batch_size, replacement = True)
    train_data_loader = torch.utils.data.DataLoader(swiss_roll_train, batch_size, num_workers = 5, shuffle = True)
    test_data_loader  = torch.utils.data.DataLoader(swiss_roll_test, batch_size, shuffle = False)

    network_info = {
        'train':{
            'z_sampler':unif,
            'train_generator':train_data_loader,
            'test_generator':test_data_loader,
            
            'lr':0.01,
            'beta1':0.9,
            'lambda':0.01,
            'lambda_exp':None,
            'eps':1e-3,
            'L':100,
            'validate':True,
            
            'epoch':10,
            'iter_per_epoch':50
        },
        'path':{
            'save_best':True,
            'save_path':'./sae_swiss_v2-test.pt',
            'tb_logs':'./tb_logs/sae_swiss'
        }
    }

    sae_swiss_v1 = SAE_swiss(network_info, device = device)
    start = time.time()
    sae_swiss_v1.train()
    print("Elapsed time: %.3fs" % (time.time() - start))