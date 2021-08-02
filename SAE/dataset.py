import torch
import PIL
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import pickle

class ToyDataset(torch.utils.data.Dataset): 
    def __init__(self, data_mat):
        self.data = data_mat
        self.z_dim = len(data_mat[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        return self.data[idx]

class CelebA(torch.utils.data.Dataset):
    def __init__(self, data_home, x_list):
        self.data_home = '%s/img_align_celeba' % data_home
        self.x_list = np.load('%s/%s' % (data_home, x_list))
        # self.list_attr = pd.read_csv('%s/list_attr_celeba.csv' % data_home)
        # self.list_bbox = pd.read_csv('%s/list_bbox_celeba.csv' % data_home)
        # self.list_eval = pd.read_csv('%s/list_eval_partition.csv' % data_home)

        # closecrop
        self.transform = transforms.Compose([
            transforms.CenterCrop((140, 140)),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),])
        
    def __len__(self):
        return len(self.x_list)
    
    def __getitem__(self, idx):
        with Image.open('%s/%s' % (self.data_home, self.x_list[idx])) as im:
            return 2.0*self.transform(im) - 1.0
        
class MNIST(torch.utils.data.Dataset):
    def __init__(self, data_home, train = True, output_channels = 1):
        self.data = None
        self.output_channels = output_channels
        if train:
            self.data = np.loadtxt('%s/mnist_train.csv' % data_home, delimiter=',', skiprows = 1)
        else:
            self.data = np.loadtxt('%s/mnist_test.csv' % data_home, delimiter=',', skiprows = 1)
        
    def __len__(self):
        return np.shape(self.data)[0]
    
    def __getitem__(self, idx):
        if self.output_channels > 1:
            return torch.from_numpy(2. * (self.data[idx, 1:785]/255) - 1.) .reshape((1,28,28)).type(torch.float32).repeat((self.output_channels,1,1))
        return torch.from_numpy(2. * (self.data[idx, 1:785]/255) - 1.) .reshape((1,28,28)).type(torch.float32)
    
    def label(self, idx):
        return self.data[idx, 0]
    
class labeled_MNIST(torch.utils.data.Dataset):
    def __init__(self, data_home, train = True, output_channels = 1):
        self.data = None
        self.output_channels = output_channels
        if train:
            self.data = np.loadtxt('%s/mnist_train.csv' % data_home, delimiter=',', skiprows = 1)
        else:
            self.data = np.loadtxt('%s/mnist_test.csv' % data_home, delimiter=',', skiprows = 1)

        self.code = np.zeros((10,10))
        for i in range(10):
            self.code[i,i] = 1.0
        self.code = torch.from_numpy(self.code).type(torch.float32)
        
    def __len__(self):
        return np.shape(self.data)[0]
    
    def __getitem__(self, idx):
        if self.output_channels > 1:
            return [torch.from_numpy(2. * (self.data[idx, 1:785]/255) - 1.).reshape((1,28,28)).type(torch.float32).repeat((self.output_channels,1,1)), self.code[self.data[idx, 0].astype(np.int)]]
        return [torch.from_numpy(2. * (self.data[idx, 1:785]/255) - 1.).reshape((1,28,28)).type(torch.float32), self.code[self.data[idx, 0].astype(np.int)]]
    
    def coded_label(self, idx):
        return self.code[idx]

class eYaleB(torch.utils.data.Dataset):
    def __init__(self, data_home, train = True, output_channels = 1):
        self.data = None
        self.output_channels = output_channels
        if train:
            with open('%s/YaleBFaceTrain.dat' % data_home, 'rb') as f:
                self.data = pickle.load(f)
        else:
            with open('%s/YaleBFaceTest.dat' % data_home, 'rb') as f:
                self.data = pickle.load(f)

        self.code = np.zeros((38,38))
        for i in range(38):
            self.code[i,i] = 1.0
        self.code = torch.from_numpy(self.code).type(torch.float32)
        
    def __len__(self):
        return (self.data['image'].shape)[0]
    
    def __getitem__(self, idx):
        if self.output_channels > 1:
            return [torch.from_numpy(2. * (self.data['image'][idx, :]/255) - 1.).reshape((1,192,168)).type(torch.float32).repeat((self.output_channels,1,1)), torch.cat((self.code[self.data['person'][idx].astype(np.int)], torch.Tensor([self.data['azimuth'][idx]/180.0]), torch.Tensor([self.data['elevation'][idx]/90.0]))).type(torch.float32)]
        return [torch.from_numpy((2. * (self.data['image'][idx, :]/255) - 1.).reshape((1,192,168))).type(torch.float32), torch.cat((self.code[self.data['person'][idx].astype(np.int)], torch.Tensor([self.data['azimuth'][idx]/180.0]), torch.Tensor([self.data['elevation'][idx]/90.0])))]

class eYaleB128(torch.utils.data.Dataset):
    def __init__(self, data_home, train = True, output_channels = 1):
        self.data = None
        self.output_channels = output_channels
        if train:
            with open('%s/YaleBFaceTrain.dat' % data_home, 'rb') as f:
                self.data = pickle.load(f)
        else:
            with open('%s/YaleBFaceTest.dat' % data_home, 'rb') as f:
                self.data = pickle.load(f)

        self.code = np.zeros((38,38))
        for i in range(38):
            self.code[i,i] = 1.0
        self.code = torch.from_numpy(self.code).type(torch.float32)
        
    def __len__(self):
        return (self.data['image'].shape)[0]
    
    def __getitem__(self, idx):
        if self.output_channels > 1:
            return [torch.from_numpy(2. * (self.data['image'][idx, :]/255) - 1.).reshape((1,128,128)).type(torch.float32).repeat((self.output_channels,1,1)), torch.cat((self.code[self.data['person'][idx].astype(np.int)], torch.Tensor([self.data['azimuth'][idx]/180.0]), torch.Tensor([self.data['elevation'][idx]/90.0]))).type(torch.float32)]
        return [torch.from_numpy((2. * (self.data['image'][idx, :]/255) - 1.).reshape((1,128,128))).type(torch.float32), torch.cat((self.code[self.data['person'][idx].astype(np.int)], torch.Tensor([self.data['azimuth'][idx]/180.0]), torch.Tensor([self.data['elevation'][idx]/90.0])))]
        