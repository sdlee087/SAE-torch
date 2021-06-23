# import os, sys, time
# import warnings
# warnings.filterwarnings("ignore")
# os.environ['CUDA_VISIBLE_DEVICES']='3'

# sys.path.append('/'.join(os.getcwd().split('/')))
# import logging_daily
# from AE_network import WAE_MMD_abstract

import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

import numpy as np
import pandas as pd

from .util import init_params, inc_avg
from .util import get_lr

class WAE_MMD_abstract(nn.Module):
    def __init__(self, network_info, log, device = 'cpu', verbose = 1):
        super(WAE_MMD_abstract, self).__init__()
        self.log = log
        if verbose == 1:
            self.log.info('------------------------------------------------------------')
            for dd in network_info['train']:
                self.log.info('%s : %s' % (dd, network_info['train'][dd]))

            for dd in network_info['path']:
                self.log.info('%s : %s' % (dd, network_info['path'][dd]))
        
        # Abstract Part. Need overriding here
        self.enc = nn.Identity()
        self.dec = nn.Identity()
        self.z_dim = 1
        
        # Concrete Part.        
        self.device = device
        self.z_sampler = network_info['train']['z_sampler'] # generate prior
        self.train_data = network_info['train']['train_data'] 
        self.test_data = network_info['train']['test_data'] 
        self.batch_size = network_info['train']['batch_size'] 
        
        self.train_generator = torch.utils.data.DataLoader(self.train_data, self.batch_size, num_workers = 5, shuffle = True, pin_memory=True, drop_last=True)
        self.test_generator = torch.utils.data.DataLoader(self.test_data, self.batch_size, num_workers = 5, shuffle = True, pin_memory=True, drop_last=True)
        
        self.save_path = network_info['path']['save_path']
        self.save_best = network_info['path']['save_best']
        self.tensorboard_dir = network_info['path']['tb_logs']
        
        self.n = self.train_generator.batch_size
        self.n_test = len(self.test_generator.dataset)
        self.validate_batch = network_info['train']['validate']
        
        self.encoder_pretrain = network_info['train']['encoder_pretrain']
        if self.encoder_pretrain:
            self.encoder_pretrain_batch_size = network_info['train']['encoder_pretrain_batch_size']
            self.encoder_pretrain_step = network_info['train']['encoder_pretrain_max_step']
            self.pretrain_generator = torch.utils.data.DataLoader(self.train_data, self.encoder_pretrain_batch_size, num_workers = 5, shuffle = True, pin_memory=True, drop_last=True)
        
        self.lr = network_info['train']['lr']
        self.beta1 = network_info['train']['beta1']
        self.lamb = network_info['train']['lambda']
        # self.lamb_exp = network_info['train']['lambda_exp']
        self.lr_schedule = network_info['train']['lr_schedule']
        
        self.num_epoch = network_info['train']['epoch']
        self.iteration = network_info['train']['iter_per_epoch']
        
        self.train_mse_list = []
        self.train_penalty_list = []
        self.test_mse_list = []
        self.test_penalty_list = []
        self.best_obj = [0, float('inf')]
        
    
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
        
    def emp_dist(self, x, y, n):
        return (self.k(x,x, False) + self.k(y,y, False))/(n*(n-1)) - 2*self.k(x,y, True)/(n*n)
        
    def forward(self, x):
        return self.dec(self.enc(x))
    
    def get_test_z(self, data):
        return self.enc(data)
    
    def generate(self, n):
        return self.enc(self.z_sampler(n, self.z_dim, device = self.device))

    def save(self, dir):
        torch.save(self.state_dict(), dir)

    def load(self, dir):
        self.load_state_dict(torch.load(dir))
        
    def pretrain_encoder(self):
        optimizer = optim.Adam(list(self.enc.parameters()), lr = self.lr, betas = (self.beta1, 0.999))
        mse = nn.MSELoss()
        
        self.log.info('------------------------------------------------------------')
        self.log.info('Pretraining Start!')
        
        cur_step = 0
        break_ind = False
        while True:
            for i, data in enumerate(self.pretrain_generator):
                cur_step = cur_step + 1
                pz = self.z_sampler(len(data), self.z_dim, device = self.device)
                x = data.to(self.device)
                qz = self.enc(x)

                qz_mean = torch.mean(qz, dim = 0)
                pz_mean = torch.mean(pz, dim = 0)

                qz_cov = torch.mean(torch.matmul((qz - qz_mean).unsqueeze(2), (qz - qz_mean).unsqueeze(1)), dim = 0)
                pz_cov = torch.mean(torch.matmul((pz - pz_mean).unsqueeze(2), (pz - pz_mean).unsqueeze(1)), dim = 0)

                loss = mse(pz_mean, qz_mean) + mse(pz_cov, qz_cov)

                loss.backward()
                optimizer.step()
                
                # train_loss_mse.append(loss.item(), len(data))
                if loss.item() > 0.1 or cur_step < 10:
                    print('train_mse: %.4f at %i step' % (loss.item(), cur_step), end = "\r")
                else:
                    self.log.info('train_mse: %.4f at %i step' % (loss.item(), cur_step))
                    break_ind = True
                    break
                    
            if break_ind or cur_step >= self.encoder_pretrain_step:
                break

    def train(self):
        self.train_mse_list = []
        self.train_penalty_list = []
        self.test_mse_list = []
        self.test_penalty_list = []
        
        self.enc.train()
        self.dec.train()
            
        if self.encoder_pretrain:
            self.pretrain_encoder()
            self.log.info('Pretraining Ended!')
            
        if self.tensorboard_dir is not None:
            self.writer = SummaryWriter(self.tensorboard_dir)
            
        mse = nn.MSELoss()
        optimizer = optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), 
                                       lr = self.lr, betas = (self.beta1, 0.999))
        self.log.info('lr : %s' % get_lr(optimizer))
        if self.lr_schedule is "manual":
            lamb = lambda e: 1.0 * (0.5 ** (e >= 30)) * (0.2 ** (e >= 50)) * (0.1 ** (e >= 100))
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lamb)

        self.log.info('------------------------------------------------------------')
        self.log.info('Training Start!')
        start_time = time.time()
        
        for epoch in range(self.num_epoch):
            # train_step
            # self.log.info('lr : %s' % get_lr(optimizer))
            train_loss_mse = inc_avg()
            train_loss_penalty = inc_avg()
            
            for i, data in enumerate(self.train_generator):
                self.enc.zero_grad()
                self.dec.zero_grad()
                
                prior_z = self.z_sampler(len(data), self.z_dim, device = self.device)
                x = data.to(self.device)

                fake_latent = self.enc(x)
                recon = self.dec(fake_latent)
                
                loss = mse(x, recon)
                if self.lamb > 0:
                    penalty = self.emp_dist(fake_latent, prior_z, self.train_generator.batch_size)
                    obj = loss + self.lamb * penalty
                else:
                    obj = loss
                obj.backward()
                optimizer.step()
                
                train_loss_mse.append(loss.item(), len(data))
                if self.lamb > 0:
                    train_loss_penalty.append(penalty.item(), len(data))
                
                print('[%i/%i]\ttrain_mse: %.4f\ttrain_penalty: %.4f' % (i+1, len(self.train_generator), train_loss_mse.avg, train_loss_penalty.avg), 
                      end = "\r")

                # if i+1 == self.iteration:
                #     break

            # print("\n", end = "\r")
            self.train_mse_list.append(train_loss_mse.avg)
            self.train_penalty_list.append(train_loss_penalty.avg)

            if self.tensorboard_dir is not None:
                self.writer.add_scalar('train/MSE', train_loss_mse.avg, epoch)
                self.writer.add_scalar('train/penalty', train_loss_penalty.avg, epoch)
                if self.hist:
                    for param_tensor in self.state_dict():
                        self.writer.add_histogram(param_tensor, self.state_dict()[param_tensor].detach().to('cpu').numpy().flatten(), epoch)
            
            # validation_step
            test_loss_mse = inc_avg()
            test_loss_penalty = inc_avg()

            if self.validate_batch:
                for i, data in enumerate(self.test_generator):
                    prior_z = self.z_sampler(len(data), self.z_dim, device = self.device).to(self.device)
                    x = data.to(self.device)
                    
                    fake_latent = self.enc(x).detach()
                    recon = self.dec(fake_latent).detach()
                    test_loss_mse.append(mse(x, recon).item(), len(data))

                    if self.lamb > 0:
                        test_loss_penalty.append(self.emp_dist(fake_latent, prior_z, self.test_generator.batch_size).item(), len(data))
                    print('[%i/%i]\ttest_mse: %.4f\ttest_penalty: %.4f' % (i, len(self.test_generator), test_loss_mse.avg, test_loss_penalty.avg), end = "\r")

                self.test_mse_list.append(test_loss_mse.avg)
                self.test_penalty_list.append(test_loss_penalty.avg)
                
                self.log.info('[%d/%d]\ttrain_mse: %.6e\ttrain_penalty: %.6e\ttest_mse: %.6e\ttest_penalty: %.6e'
                      % (epoch + 1, self.num_epoch, train_loss_mse.avg, train_loss_penalty.avg, test_loss_mse.avg, test_loss_penalty.avg))
                # print('[%d/%d]\ttrain_mse: %.6e\ttrain_penalty: %.6e\ttest_mse: %.6e\ttest_penalty: %.6e'
                #       % (epoch + 1, self.num_epoch, train_loss_mse.avg, train_loss_penalty.avg, test_loss_mse.avg, test_loss_penalty.avg))

                if self.tensorboard_dir is not None:
                    self.writer.add_scalar('test/MSE', test_loss_mse.avg, epoch)
                    self.writer.add_scalar('test/penalty', test_loss_penalty.avg, epoch)
                    
                    prior_z = self.z_sampler(self.test_generator.batch_size, self.z_dim, device = self.device)
                    data = next(iter(self.test_generator))
                    x = data.to(self.device)
                    fake_latent = self.enc(x).detach()
                    recon = self.dec(fake_latent).detach()

                    if self.lamb > 0:
                        # Embedding
                        for_embed1 = fake_latent.to('cpu').numpy()
                        for_embed2 = prior_z.to('cpu').numpy()
                        label = ['fake']*len(for_embed1) + ['prior']*len(for_embed2)
                        self.writer.add_embedding(np.concatenate((for_embed1, for_embed2)), metadata = label, global_step = epoch)

                        # Sample Generation
                        test_dec = self.dec(prior_z).detach().to('cpu').numpy()
                        self.writer.add_images('generated_sample', (test_dec[0:32])*0.5 + 0.5, epoch)

                    # Reconstruction
                    self.writer.add_images('reconstruction', (np.concatenate((x.to('cpu').numpy()[0:16], recon.to('cpu').numpy()[0:16])))*0.5 + 0.5, epoch)
                    self.writer.flush()
                    
                
                if self.save_best:
                    obj = test_loss_mse.avg + self.lamb * test_loss_penalty.avg
                    if self.best_obj[1] > obj:
                        self.best_obj[0] = epoch + 1
                        self.best_obj[1] = obj
                        self.save(self.save_path)
                        self.log.info("model saved, obj: %.6e" % obj)
                else:
                    self.save(self.save_path)
                    # self.log.info("model saved at: %s" % self.save_path)
                        
            # if self.lamb_exp is not None:
            #    self.lamb = self.lamb_exp * self.lamb
                
            if self.lr_schedule is not None:
                scheduler.step()
            
        if not self.validate_batch:
            self.save(self.save_path)
            # self.log.info("model saved at: %s" % self.save_path)

        self.log.info('Training Finished!')
        self.log.info("Elapsed time: %.3fs" % (time.time() - start_time))

        if self.tensorboard_dir is not None:
            self.writer.close()
    
    # def train(self):
    #     self.train_mse_list = []
    #     self.train_penalty_list = []
    #     self.test_mse_list = []
    #     self.test_penalty_list = []
        
    #     mse = nn.MSELoss()
    #     optimizer = optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), 
    #                                    lr = self.lr, betas = (self.beta1, 0.999))
        
    #     self.enc.train()
    #     self.dec.train()
        
    #     for epoch in range(self.num_epoch):
    #         # train_step
    #         train_loss_mse = inc_avg()
    #         train_loss_penalty = inc_avg()
            
    #         for i in range(self.iteration):
    #             data = next(iter(self.train_generator))
    #             self.enc.zero_grad()
    #             self.dec.zero_grad()
                
    #             prior_z = self.z_sampler(len(data), self.z_dim, device = self.device).to(self.device)
    #             x = data.to(self.device)

    #             fake_latent = self.enc(x)
    #             recon = self.dec(fake_latent)
                
    #             loss = mse(x, recon)
    #             penalty = self.emp_dist(fake_latent, prior_z, self.train_generator.batch_size)

    #             obj = loss + self.lamb * penalty
    #             obj.backward()
    #             optimizer.step()
                
    #             train_loss_mse.append(loss.item(), len(data))
    #             train_loss_penalty.append(penalty.item(), len(data))
                
    #             print('[%i/%i]\ttrain_mse: %.6e\ttrain_penalty: %.6e' % (i+1, self.iteration, train_loss_mse.avg, train_loss_penalty.avg), 
    #                   end = "\r")
                
    #         self.train_mse_list.append(train_loss_mse.avg)
    #         self.train_penalty_list.append(train_loss_penalty.avg)
            
    #         # validation_step
    #         test_loss_mse = inc_avg()
    #         test_loss_penalty = inc_avg()
    #         if self.validate_batch:
    #             for data in self.test_generator:
    #                 prior_z = self.z_sampler(len(data), self.z_dim, device = self.device).to(self.device)
    #                 x = data.to(self.device)
                    
    #                 fake_latent = self.enc(x).detach()
    #                 recon = self.dec(fake_latent).detach()
                    
    #                 test_loss_mse.append(mse(x, recon).item(), len(data))
    #                 test_loss_penalty.append(self.emp_dist(fake_latent, prior_z, self.test_generator.batch_size).item(), len(data))
                
    #             self.test_mse_list.append(test_loss_mse.avg)
    #             self.test_penalty_list.append(test_loss_penalty.avg)
            
    #             print('[%d/%d]\ttrain_mse: %.6e\ttrain_penalty: %.6e\ttest_mse: %.6e\ttest_penalty: %.6e'
    #                   % (epoch + 1, self.num_epoch, train_loss_mse.avg, train_loss_penalty.avg, test_loss_mse.avg, test_loss_penalty.avg))
                
    #             if self.save_best:
    #                 obj = test_loss_mse.avg + self.lamb * test_loss_penalty.avg
    #                 if self.best_obj[1] > obj:
    #                     self.best_obj[0] = epoch + 1
    #                     self.best_obj[1] = obj
    #                     self.save(self.save_path)
    #                     print("model saved, obj: %.6e" % obj)
                        
    #         if self.lamb_exp is not None:
    #             self.lamb = self.lamb_exp * self.lamb
            
    #     if not self.validate_batch:
    #         self.save(self.save_path)
    #         print("model saved at: %s" % self.save_path)
    #     elif not self.save_best:
    #         self.save(self.save_path)
    #         print("model saved at: %s" % self.save_path)

            
class WAE_MMD_swiss(WAE_MMD_abstract):
    def __init__(self, network_info, log, device = 'cpu', verbose = 1):
        super(WAE_MMD_swiss, self).__init__(network_info, log, device, verbose)
    
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
    
class SAE_abstract(WAE_MMD_abstract):
    def __init__(self, network_info, log, device = 'cpu', verbose = 1):
        super(SAE_abstract, self).__init__(network_info, log, device, verbose)

        self.eps = network_info['train']['eps']
        self.L = network_info['train']['L']
    
    def sinkhorn(self, x, y):
        C = ((x.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(dim = 2))/self.eps
        
        def M(u,v):
            return (-C + u.unsqueeze(1) + v.unsqueeze(0))/self.eps
        
        def lse(A):
            return A.logsumexp(dim = 1, keepdims = True)
        
        u = torch.zeros_like(C[0])
        v = torch.zeros_like(C[0])
        
        for i in range(self.L):
            u = self.eps*(-lse(M(u,v)).squeeze(1)) + u
            v = self.eps*(-lse(M(u,v).transpose(0, 1)).squeeze(1)) + v
            
        u_final, v_final = u, v
        pi = M(u_final, v_final).exp()
        cost = (pi*C).sum()
        return cost
    
    def emp_dist(self, x, y, n):
        return (self.sinkhorn(x,y) - (self.sinkhorn(x,x) + self.sinkhorn(y,y))/2.0)/n
    
    
    def get_penalty(self):
        for data in self.test_generator:
            prior_z = self.z_sampler(len(data), self.z_dim, device = self.device)
            x = data.to(device)
            fake_latent = self.enc(x).detach()

            return self.emp_dist(fake_latent, prior_z, self.test_generator.batch_size).item()
        
    
class SAE_swiss(SAE_abstract):
    def __init__(self, network_info, log, device = 'cpu', verbose = 1):
        super(SAE_swiss, self).__init__(network_info, log, device, verbose)

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

