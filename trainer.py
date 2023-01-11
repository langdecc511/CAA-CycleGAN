import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import time
import torch
import datetime

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from sagan_models import Generator, Discriminator
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import make_folder


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class logcosh(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, true, pred):
        loss = torch.log(torch.cosh(pred - true))
        return torch.sum(loss)




class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal1(m):
    if isinstance(m, nn.Conv1d):
        m.weight.data.normal_(0, 0.1)
        m.bias.data.zero_()
    elif isinstance(m, nn.InstanceNorm1d):
        # pass
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias, 0)
        
def weights_init_normal(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.InstanceNorm1d):
        pass
        # nn.init.constant_(m.weight,1)
        # nn.init.constant_(m.bias, 0)        
        
        


class Trainer(object):
    def __init__(self, data_loader, config):
        self.data_loader = data_loader
        self.model = config.model
        self.adv_loss = config.adv_loss

        # Model hyper-parameters
        self.imsize = config.imsize
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.parallel = config.parallel

        self.lambda_gp = config.lambda_gp
        self.total_step = config.total_step
        self.d_iters = config.d_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_AECG_lr = config.g_AECG_lr
        self.g_MECG_lr = config.g_MECG_lr
        self.g_FECG_lr = config.g_FECG_lr
        self.g_BIAS_lr = config.g_BIAS_lr

        self.d_AECG_lr = config.d_AECG_lr
        self.d_MECG_lr = config.d_MECG_lr
        self.d_FECG_lr = config.d_FECG_lr
        self.d_BIAS_lr = config.d_BIAS_lr
        
        
        self.decay_start_epoch = round(config.total_step / config.batch_size) - 1
        self.decay_start_epoch = 1
        
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.dataset = config.dataset
        self.use_tensorboard = config.use_tensorboard
        self.image_path = config.image_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version
        

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        self.build_model()
        
        
        
        self.gamma_FECG_fake  = nn.Parameter(torch.zeros(1)).to(device)
        self.lambda_MECG_fake = nn.Parameter(torch.zeros(1)).to(device)
        self.beta_BIAS_fake   = nn.Parameter(torch.zeros(1)).to(device)
        
        self.gamma_FECG_reconstr  = nn.Parameter(torch.zeros(1)).to(device)
        self.lambda_MECG_reconstr = nn.Parameter(torch.zeros(1)).to(device)
        self.beta_BIAS_reconstr   = nn.Parameter(torch.zeros(1)).to(device)
        
        self.gamma_AECG_loss = nn.Parameter(torch.ones(1)*0.8).to(device)
        self.gamma_FECG_loss = nn.Parameter(torch.ones(1)*0.4).to(device)
        self.gamma_MECG_loss = nn.Parameter(torch.ones(1)*0.4).to(device)
        self.gamma_BIAS_loss = nn.Parameter(torch.ones(1)*0.2).to(device)
        

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()



    def train(self):
        model_save_step = int(self.model_save_step)

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0


        #  Train Discriminators
        self.model_train()
        # Start time
        start_time = time.time()
        loss_c = 10000
        MECG_factor = 5.0
        FECG_factor = 5.0
        BIAS_factor = 0.5
        for step in range(start, self.total_step):
            tbar = tqdm(self.data_loader, desc='epoch'+str(step))
            for AECG_signals, FECG_signals, MECG_signals,BIAS_signals in tbar: 
                # print(AECG_signals.shape)
                # MECG_signals =  AECG_signals - FECG_signals   
                # BIAS_signals = tensor2var(torch.randn(AECG_signals.size(0), AECG_signals.size(1), AECG_signals.size(2))*torch.max(AECG_signals)*0.1).expand_as(AECG_signals)
                
                
                # print('A',AECG_signals,'\n')
                # print('\n')
                # print('F',FECG_signals,'\n')
                # print('\n')
                # print('M',MECG_signals,'\n')
                # print('\n')
                # print('B',BIAS_signals,'\n')
                # print('\n')
                # BIAS_signals = AECG_signals - FECG_signals - MECG_signals
                valid = torch.ones((AECG_signals.shape[0],1,128),dtype=torch.float32).to(device)
                fake = torch.zeros((AECG_signals.shape[0],1,128),dtype=torch.float32).to(device)
                # valid = Variable(torch.Tensor((AECG_signals.shape[0],1,1)).fill_(1.0).float(), requires_grad=False).to(device)
                # fake =  Variable(torch.Tensor((AECG_signals.shape[0],1,1)).fill_(0.0).float(), requires_grad=False).to(device)
                
                
                AECG_signals = AECG_signals.to(device)
                FECG_signals = FECG_signals.to(device)
                MECG_signals = MECG_signals.to(device)
                BIAS_signals = BIAS_signals.to(device)
                # plt.plot(AECG_signals[0].t().cpu().numpy(),'r')    
                # plt.show()
                # plt.plot(FECG_signals[0].t().cpu().numpy(),'g')
                # plt.show()
                # plt.plot(MECG_signals[0][0].detach().cpu().numpy(),'b')
                # plt.title('MECG')
                # plt.show()
                


                self.optimizer_G_zero_grad()
                #AECG to MECG
                #1 generator loss
                # fake_MECG_signals = self.G_AECG2MECG(AECG_signals)
                # fake_AECG_signals = self.G_MECG2AECG(MECG_signals)
                
                # reco_AECG_signals = self.G_MECG2AECG(fake_MECG_signals)
                # reco_MECG_signals = self.G_AECG2MECG(fake_AECG_signals)
                
                # d_AECG_signals = self.D_AECG2MECG(AECG_signals)
                # d_AECG_loss_real = self.loss_discriminator(d_AECG_signals,valid)    
                # d_fake_AECG_signals = self.D_AECG2MECG(fake_AECG_signals)
                # d_AECG_loss_fake = self.loss_discriminator(d_fake_AECG_signals,fake)  
                # d_AECG_loss = (d_AECG_loss_real + d_AECG_loss_fake)*0.5 
                
                # d_MECG_signals = self.D_MECG2AECG(MECG_signals)
                # d_MECG_loss_real = self.loss_discriminator(d_MECG_signals,valid)
                # d_fake_MECG_signals = self.D_MECG2AECG(fake_MECG_signals)
                # d_MECG_loss_fake = self.loss_discriminator(d_fake_MECG_signals,fake) 
                # d_MECG_loss = (d_MECG_loss_real + d_MECG_loss_fake)*0.5 
                
                # d_AECG2MECG_loss = (d_AECG_loss + d_MECG_loss)*0.5
                
                
                
                
                same_MECG_signals = self.G_AECG2MECG(AECG_signals)
                loss_generator_MECG = self.loss_generator(same_MECG_signals,MECG_signals.float())*1             
                same_AECG_signals = self.G_MECG2AECG(MECG_signals)   
                loss_generator_AECG = self.loss_generator(same_AECG_signals,AECG_signals.float())*1
                
                
                #2 forwardGAN loss
                fake_MECG_signals = self.G_AECG2MECG(AECG_signals)
                pred_fake_MECG_signals = self.D_AECG2MECG(fake_MECG_signals)
                loss_forwardGAN_AECG2MECG = self.loss_forwardGAN(pred_fake_MECG_signals,valid)
                
                fake_AECG_signals_from_MECG = self.G_MECG2AECG(MECG_signals)
                pred_fake_AECG_signals_from_MECG = self.D_MECG2AECG(fake_AECG_signals_from_MECG)
                loss_forwardGAN_MECG2AECG = self.loss_forwardGAN(pred_fake_AECG_signals_from_MECG,valid)
                
                #3 cycleGAN loss
                reconstr_AECG_signals_from_MECG = self.G_MECG2AECG(fake_MECG_signals)
                loss_cycleGAN_AECG2MECG2AECG = self.loss_cycleGAN(reconstr_AECG_signals_from_MECG,AECG_signals.float())*0.04
                
                reconstr_MECG_signals = self.G_AECG2MECG(fake_AECG_signals_from_MECG)
                loss_cycleGAN_MECG2AECG2MECG = self.loss_cycleGAN(reconstr_MECG_signals,MECG_signals.float())*0.04
                
                loss_G_total_AECG2MECG = loss_generator_MECG + loss_generator_AECG + loss_forwardGAN_AECG2MECG + loss_forwardGAN_MECG2AECG + loss_cycleGAN_AECG2MECG2AECG + loss_cycleGAN_MECG2AECG2MECG
                loss_G_total_AECG2MECG.backward(retain_graph=True)
                
                
                
                
                #AECG to FECG
                #1 generator loss
                same_FECG_signals = self.G_AECG2FECG(AECG_signals)
                loss_generator_FECG = self.loss_generator(same_FECG_signals,FECG_signals.float())*4               
                same_AECG_signals = self.G_FECG2AECG(FECG_signals)   
                loss_generator_AECG = self.loss_generator(same_AECG_signals,AECG_signals.float())*4
                
                
                #2 forwardGAN loss
                fake_FECG_signals = self.G_AECG2FECG(AECG_signals)
                pred_fake_FECG_signals = self.D_AECG2FECG(fake_FECG_signals)
                loss_forwardGAN_AECG2FECG = self.loss_forwardGAN(pred_fake_FECG_signals,valid)
                
                fake_AECG_signals_from_FECG = self.G_FECG2AECG(FECG_signals)
                pred_fake_AECG_signals_from_FECG = self.D_FECG2AECG(fake_AECG_signals_from_FECG)
                loss_forwardGAN_FECG2AECG = self.loss_forwardGAN(pred_fake_AECG_signals_from_FECG,valid)
                
                #3 cycleGAN loss
                reconstr_AECG_signals_from_FECG = self.G_FECG2AECG(fake_FECG_signals)
                loss_cycleGAN_AECG2FECG2AECG = self.loss_cycleGAN(reconstr_AECG_signals_from_FECG,AECG_signals.float())*0.04
                
                reconstr_FECG_signals = self.G_AECG2FECG(fake_AECG_signals_from_FECG)
                loss_cycleGAN_FECG2AECG2FECG = self.loss_cycleGAN(reconstr_FECG_signals,FECG_signals.float())*0.04
                
                loss_G_total_AECG2FECG = loss_generator_FECG + loss_generator_AECG + loss_forwardGAN_AECG2FECG + loss_forwardGAN_FECG2AECG + loss_cycleGAN_AECG2FECG2AECG + loss_cycleGAN_FECG2AECG2FECG
                loss_G_total_AECG2FECG.backward(retain_graph=True)

                
                #AECG to BIAS
                #1 generator loss
                same_BIAS_signals = self.G_AECG2BIAS(AECG_signals)
                loss_generator_BIAS = self.loss_generator(same_BIAS_signals,BIAS_signals.float())*1               
                same_AECG_signals = self.G_BIAS2AECG(BIAS_signals)   
                loss_generator_AECG = self.loss_generator(same_AECG_signals,AECG_signals.float())*1
                
                
                #2 forwardGAN loss
                fake_BIAS_signals = self.G_AECG2BIAS(AECG_signals)
                pred_fake_BIAS_signals = self.D_AECG2BIAS(fake_BIAS_signals)
                loss_forwardGAN_AECG2BIAS = self.loss_forwardGAN(pred_fake_BIAS_signals,valid)
                
                fake_AECG_signals_from_BIAS = self.G_BIAS2AECG(BIAS_signals)
                pred_fake_AECG_signals_from_BIAS = self.D_BIAS2AECG(fake_AECG_signals_from_BIAS)
                loss_forwardGAN_BIASAECG = self.loss_forwardGAN(pred_fake_AECG_signals_from_BIAS,valid)
                
                #3 cycleGAN loss
                reconstr_AECG_signals_from_BIAS = self.G_BIAS2AECG(fake_BIAS_signals)
                loss_cycleGAN_AECG2BIAS2AECG = self.loss_cycleGAN(reconstr_AECG_signals_from_BIAS,AECG_signals.float())*0.04
                
                reconstr_BIAS_signals = self.G_AECG2BIAS(fake_AECG_signals_from_BIAS)
                loss_cycleGAN_BIAS2AECG2BIAS = self.loss_cycleGAN(reconstr_BIAS_signals,BIAS_signals.float())*0.04
                
                loss_G_total_AECG2BIAS = loss_generator_BIAS + loss_generator_AECG + loss_forwardGAN_AECG2BIAS + loss_forwardGAN_BIASAECG + loss_cycleGAN_AECG2BIAS2AECG + loss_cycleGAN_BIAS2AECG2BIAS
                loss_G_total_AECG2BIAS.backward(retain_graph=True) 
                

                
                
                
                
                #D loss      
                self.optimizer_D_zero_grad()  
                #AECG to MECG
                pred_MECG_signals = self.D_AECG2MECG(AECG_signals)
                loss_D_real_forwardGAN_AECG2MECG = self.loss_forwardGAN(pred_MECG_signals,valid)
                pred_fake_MECG_signals = self.D_AECG2MECG(fake_AECG_signals_from_MECG)
                loss_D_fake_forwardGAN_AECG2MECG = self.loss_forwardGAN(pred_fake_MECG_signals, fake)               
                loss_D_forwardGAN_AECG2MECG= (loss_D_real_forwardGAN_AECG2MECG +  loss_D_fake_forwardGAN_AECG2MECG)*0.5
                loss_D_forwardGAN_AECG2MECG.backward(retain_graph=True)
                
                pred_AECG_signals = self.D_MECG2AECG(MECG_signals)
                loss_D_real_forwardGAN_MECG2AECG = self.loss_forwardGAN(pred_AECG_signals,valid)
                pred_fake_AECG_signals_from_MECG = self.D_MECG2AECG(fake_MECG_signals)
                loss_D_fake_forwardGAN_MECG2AECG = self.loss_forwardGAN(pred_fake_AECG_signals_from_MECG, fake)               
                loss_D_forwardGAN_MECG2AECG= (loss_D_real_forwardGAN_MECG2AECG +  loss_D_fake_forwardGAN_MECG2AECG)*0.5

                
                loss_D_AECG2MECG = (loss_D_forwardGAN_AECG2MECG + loss_D_forwardGAN_MECG2AECG)*0.5
                loss_D_AECG2MECG.backward(retain_graph=True)
                

                #AECG to FECG
                pred_FECG_signals = self.D_AECG2FECG(AECG_signals)
                loss_D_real_forwardGAN_AECG2FECG = self.loss_forwardGAN(pred_FECG_signals,valid)
                pred_fake_FECG_signals = self.D_AECG2FECG(fake_AECG_signals_from_FECG)
                loss_D_fake_forwardGAN_AECG2FECG = self.loss_forwardGAN(pred_fake_FECG_signals, fake)               
                loss_D_forwardGAN_AECG2FECG= (loss_D_real_forwardGAN_AECG2FECG +  loss_D_fake_forwardGAN_AECG2FECG)*0.5
                loss_D_forwardGAN_AECG2FECG.backward(retain_graph=True)
                
                pred_AECG_signals = self.D_FECG2AECG(FECG_signals)
                loss_D_real_forwardGAN_FECG2AECG = self.loss_forwardGAN(pred_AECG_signals,valid)
                pred_fake_AECG_signals_from_FECG = self.D_FECG2AECG(fake_FECG_signals)
                loss_D_fake_forwardGAN_FECG2AECG = self.loss_forwardGAN(pred_fake_AECG_signals_from_FECG, fake)               
                loss_D_forwardGAN_FECG2AECG= (loss_D_real_forwardGAN_FECG2AECG +  loss_D_fake_forwardGAN_FECG2AECG)*0.5

                
                loss_D_AECG2FECG = (loss_D_forwardGAN_AECG2FECG + loss_D_forwardGAN_FECG2AECG)*0.5
                loss_D_AECG2FECG.backward(retain_graph=True)
                
                
                #AECG to BIAS
                pred_BIAS_signals = self.D_AECG2BIAS(AECG_signals)
                loss_D_real_forwardGAN_AECG2BIAS = self.loss_forwardGAN(pred_BIAS_signals,valid)
                pred_fake_BIAS_signals = self.D_AECG2BIAS(fake_AECG_signals_from_BIAS)
                loss_D_fake_forwardGAN_AECG2BIAS = self.loss_forwardGAN(pred_fake_BIAS_signals, fake)               
                loss_D_forwardGAN_AECG2BIAS= (loss_D_real_forwardGAN_AECG2BIAS +  loss_D_fake_forwardGAN_AECG2BIAS)*0.5
                loss_D_forwardGAN_AECG2BIAS.backward(retain_graph=True)
                
                pred_AECG_signals = self.D_BIAS2AECG(BIAS_signals)
                loss_D_real_forwardGAN_BIAS_AECG = self.loss_forwardGAN(pred_AECG_signals,valid)
                pred_fake_AECG_signals_from_BIAS = self.D_BIAS2AECG(fake_BIAS_signals)
                loss_D_fake_forwardGAN_BIAS2AECG = self.loss_forwardGAN(pred_fake_AECG_signals_from_BIAS, fake)               
                loss_D_forwardGAN_BIAS2AECG= (loss_D_real_forwardGAN_BIAS_AECG +  loss_D_fake_forwardGAN_BIAS2AECG)*0.5

                
                loss_D_AECG2BIAS = (loss_D_forwardGAN_AECG2BIAS + loss_D_forwardGAN_BIAS2AECG)*0.5
                loss_D_AECG2BIAS.backward(retain_graph=True)
                
                
                self.optimizer_G_step()  
                self.optimizer_D_step() 
                
            self.optimizer_G_lr_step()
            self.optimizer_D_lr_step()
                


            tbar.close()
            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], D_FECG_loss: {:.6f}, ".
                      format(elapsed, step + 1, self.total_step, (step + 1),
                              self.total_step , 
                              loss_generator_FECG.item() ))

                # print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], G_FECG_loss: {:.6f}, G_FECG_lr: {:.6f}, D_FECG_lr: {:.6f}, "
                #       " G_AECG2FECG_ccar03_ave_gamma: {:.4f}, G_AECG2FECG_ccar02_ave_gamma: {:.4f}".
                #       format(elapsed, step + 1, self.total_step, (step + 1),
                #               self.total_step , 
                #               loss_generator_FECG.item(),self.G_AECG2FECG_exp_lr_scheduler.get_last_lr()[0], self.D_AECG2FECG_exp_lr_scheduler.get_last_lr()[0],
                #               self.G_AECG2FECG.module.ccar03.gamma.mean().item(), self.G_AECG2FECG.module.ccar02.gamma.mean().item() ))

            # Sample images
            if (step + 1) % self.sample_step == 0:
                fake_FECG_signals = self.G_AECG2FECG(AECG_signals)
                self.sample_images(epoch=step, batch_i=step, MECG=denorm(AECG_signals.cpu().detach().numpy()), FECG_reconstr=denorm(fake_FECG_signals.cpu().detach().numpy()),FECG=denorm(FECG_signals.cpu().detach().numpy()), sample_path=self.sample_path)


        
            # if (step+1) % model_save_step==0:
            if loss_c > loss_generator_FECG.item():
                loss_c = loss_generator_FECG.item()
                torch.save(self.G_AECG2MECG.state_dict(),os.path.join(self.model_save_path, '{}_G_AECG2MECG.pth'.format(step + 1)))
                torch.save(self.G_MECG2AECG.state_dict(),os.path.join(self.model_save_path, '{}_G_MECG2AECG.pth'.format(step + 1)))
                torch.save(self.D_AECG2MECG.state_dict(),os.path.join(self.model_save_path, '{}_D_AECG2MECG.pth'.format(step + 1)))
                torch.save(self.D_MECG2AECG.state_dict(),os.path.join(self.model_save_path, '{}_D_MECG2AECG.pth'.format(step + 1)))
                
                
                torch.save(self.G_AECG2FECG.state_dict(),os.path.join(self.model_save_path, '{}_G_AECG2FECG.pth'.format(step + 1)))
                torch.save(self.G_AECG2FECG.state_dict(),os.path.join(self.model_save_path, '{}_G_AECG2FECG.pth'.format(step + 1)))
                torch.save(self.D_AECG2FECG.state_dict(),os.path.join(self.model_save_path, '{}_D_AECG2FECG.pth'.format(step + 1)))
                torch.save(self.D_FECG2AECG.state_dict(),os.path.join(self.model_save_path, '{}_D_FECG2AECG.pth'.format(step + 1)))
                
                torch.save(self.G_AECG2BIAS.state_dict(),os.path.join(self.model_save_path, '{}_G_AECG2BIAS.pth'.format(step + 1)))
                torch.save(self.G_BIAS2AECG.state_dict(),os.path.join(self.model_save_path, '{}_G_BIAS2AECG.pth'.format(step + 1)))
                torch.save(self.D_AECG2BIAS.state_dict(),os.path.join(self.model_save_path, '{}_D_AECG2BIAS.pth'.format(step + 1)))
                torch.save(self.D_BIAS2AECG.state_dict(),os.path.join(self.model_save_path, '{}_D_BIAS2AECG.pth'.format(step + 1)))
                
                

    def build_model(self):
        
        #first, create generator 
        
        #AECG to MECG
        self.G_AECG2MECG = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).to(device)
        self.G_MECG2AECG = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).to(device)     
        self.D_AECG2MECG = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).to(device)
        self.D_MECG2AECG = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).to(device)
        
        # self.G_AECG2MECG = nn.DataParallel(self.G_AECG2MECG)
        # self.G_MECG2AECG = nn.DataParallel(self.G_MECG2AECG)
        # self.D_AECG2MECG = nn.DataParallel(self.D_AECG2MECG)
        # self.D_MECG2AECG = nn.DataParallel(self.D_MECG2AECG)

        
        #AECG to FECG
        self.G_AECG2FECG = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).to(device)
        self.G_FECG2AECG = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).to(device)
        self.D_AECG2FECG = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).to(device)
        self.D_FECG2AECG = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).to(device)
        
        # self.G_AECG2FECG = nn.DataParallel(self.G_AECG2FECG)
        # self.G_FECG2AECG = nn.DataParallel(self.G_FECG2AECG)
        # self.D_AECG2FECG = nn.DataParallel(self.D_AECG2FECG)
        # self.D_FECG2AECG = nn.DataParallel(self.D_FECG2AECG)
        
        #AECG to BIAS
        self.G_AECG2BIAS = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).to(device)
        self.G_BIAS2AECG = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).to(device)
        self.D_AECG2BIAS = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).to(device)
        self.D_BIAS2AECG = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).to(device)
        
        # self.G_AECG2BIAS = nn.DataParallel(self.G_AECG2BIAS)
        # self.G_BIAS2AECG = nn.DataParallel(self.G_BIAS2AECG)
        # self.D_AECG2BIAS = nn.DataParallel(self.D_AECG2BIAS)
        # self.D_BIAS2AECG = nn.DataParallel(self.D_BIAS2AECG)
        
        
        #second, initialize weights 
        # self.G_AECG2MECG.apply(weights_init_normal)
        # self.G_MECG2AECG.apply(weights_init_normal)    
        # self.D_AECG2MECG.apply(weights_init_normal)
        # self.D_MECG2AECG.apply(weights_init_normal)

        # self.G_AECG2FECG.apply(weights_init_normal)
        # self.G_FECG2AECG.apply(weights_init_normal)
        # self.D_AECG2FECG.apply(weights_init_normal)
        # self.D_FECG2AECG.apply(weights_init_normal)

        # self.G_AECG2BIAS.apply(weights_init_normal)
        # self.G_BIAS2AECG.apply(weights_init_normal)
        # self.D_AECG2BIAS.apply(weights_init_normal)
        # self.D_BIAS2AECG.apply(weights_init_normal)
        
        #third, loss definition
        # self.loss_generator = torch.nn.L1Loss()
        # self.loss_forwardGAN = torch.nn.MSELoss() 
        # self.loss_cycleGAN = torch.nn.L1Loss()
        
        self.loss_generator = logcosh()
        self.loss_forwardGAN = torch.nn.L1Loss() 
        self.loss_cycleGAN = logcosh()
        
        
        # self.loss_discriminator = torch.nn.L1Loss()

        #fourth, optimizer definition
        self.G_AECG2MECG_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G_AECG2MECG.parameters()), self.g_AECG_lr, [self.beta1, self.beta2])        
        self.G_MECG2AECG_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G_MECG2AECG.parameters()), self.g_MECG_lr, [self.beta1, self.beta2])
        self.D_AECG2MECG_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_AECG2MECG.parameters()), self.d_AECG_lr, [self.beta1, self.beta2])
        self.D_MECG2AECG_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_MECG2AECG.parameters()), self.d_MECG_lr, [self.beta1, self.beta2])
  
        self.G_AECG2FECG_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G_AECG2FECG.parameters()), self.g_AECG_lr, [self.beta1, self.beta2])        
        self.G_FECG2AECG_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G_FECG2AECG.parameters()), self.g_FECG_lr, [self.beta1, self.beta2])
        self.D_AECG2FECG_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_AECG2FECG.parameters()), self.d_AECG_lr, [self.beta1, self.beta2])
        self.D_FECG2AECG_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_FECG2AECG.parameters()), self.d_FECG_lr, [self.beta1, self.beta2])
    
        self.G_AECG2BIAS_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G_AECG2BIAS.parameters()), self.g_AECG_lr, [self.beta1, self.beta2])        
        self.G_BIAS2AECG_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G_BIAS2AECG.parameters()), self.g_BIAS_lr, [self.beta1, self.beta2])
        self.D_AECG2BIAS_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_AECG2BIAS.parameters()), self.d_AECG_lr, [self.beta1, self.beta2])
        self.D_BIAS2AECG_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_BIAS2AECG.parameters()), self.d_BIAS_lr, [self.beta1, self.beta2])

        # print(self.decay_start_epoch)
        #fifth, lr_scheduler definition
 
        self.G_AECG2MECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.G_AECG2MECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.G_MECG2AECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.G_MECG2AECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.D_AECG2MECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.D_AECG2MECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.D_MECG2AECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.D_MECG2AECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        
        self.G_AECG2FECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.G_AECG2FECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.G_FECG2AECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.G_FECG2AECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.D_AECG2FECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.D_AECG2FECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.D_FECG2AECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.D_FECG2AECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        
        self.G_AECG2BIAS_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.G_AECG2BIAS_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.G_BIAS2AECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.G_BIAS2AECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.D_AECG2BIAS_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.D_AECG2BIAS_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.D_BIAS2AECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.D_BIAS2AECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        
    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def optimizer_G_zero_grad(self):
        self.G_AECG2MECG_optimizer.zero_grad()
        self.G_MECG2AECG_optimizer.zero_grad()
        self.G_AECG2FECG_optimizer.zero_grad()
        self.G_FECG2AECG_optimizer.zero_grad()
        self.G_AECG2BIAS_optimizer.zero_grad()
        self.G_BIAS2AECG_optimizer.zero_grad()
        
    def optimizer_D_zero_grad(self):    
        self.D_AECG2MECG_optimizer.zero_grad()
        self.D_MECG2AECG_optimizer.zero_grad()
        self.D_AECG2FECG_optimizer.zero_grad()
        self.D_FECG2AECG_optimizer.zero_grad()
        self.D_AECG2BIAS_optimizer.zero_grad()
        self.D_BIAS2AECG_optimizer.zero_grad()
        
    def optimizer_G_step(self):
        self.G_AECG2MECG_optimizer.step()
        self.G_MECG2AECG_optimizer.step()
        self.G_AECG2FECG_optimizer.step()
        self.G_FECG2AECG_optimizer.step()
        self.G_AECG2BIAS_optimizer.step()
        self.G_BIAS2AECG_optimizer.step()
        
    def optimizer_D_step(self):    
        self.D_AECG2MECG_optimizer.step()
        self.D_MECG2AECG_optimizer.step()
        self.D_AECG2FECG_optimizer.step()
        self.D_FECG2AECG_optimizer.step()
        self.D_AECG2BIAS_optimizer.step()
        self.D_BIAS2AECG_optimizer.step()
        
        
        

    def optimizer_G_lr_step(self):     
        self.G_AECG2MECG_exp_lr_scheduler.step()
        self.G_MECG2AECG_exp_lr_scheduler.step()
        self.G_AECG2FECG_exp_lr_scheduler.step()
        self.G_FECG2AECG_exp_lr_scheduler.step()
        self.G_AECG2BIAS_exp_lr_scheduler.step()
        self.G_BIAS2AECG_exp_lr_scheduler.step()

    def optimizer_D_lr_step(self):
        self.D_AECG2MECG_exp_lr_scheduler.step()
        self.D_MECG2AECG_exp_lr_scheduler.step() 
        self.D_AECG2FECG_exp_lr_scheduler.step()
        self.D_FECG2AECG_exp_lr_scheduler.step()
        self.D_AECG2BIAS_exp_lr_scheduler.step()
        self.D_BIAS2AECG_exp_lr_scheduler.step()


      
        
    def model_train(self):
        self.G_AECG2MECG.train()
        self.G_MECG2AECG.train()    
        self.D_AECG2MECG.train()
        self.D_MECG2AECG.train()
        
        self.G_AECG2FECG.train()
        self.G_FECG2AECG.train()
        self.D_AECG2FECG.train()
        self.D_FECG2AECG.train()
    
        self.G_AECG2BIAS.train()
        self.G_BIAS2AECG.train()
        self.D_AECG2BIAS.train()
        self.D_BIAS2AECG.train()
        

            

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
        
        
    def sample_images(self, epoch, batch_i, MECG, FECG_reconstr, FECG, sample_path):
        r, c = 1, 3
        gen_imgs = [MECG, FECG_reconstr,FECG]
        titles = ['MECG', 'FECG_reconstr','FECG']
        

        fig, axs = plt.subplots(r, c,figsize=(15, 5))
        cnt = 0
        for i in range(r):
            for j in range(c):
                for bias in range(1):
                    tt = gen_imgs[cnt][bias,:]
                    axs[j].plot(tt[0])
                axs[j].set_title(titles[j])
                cnt += 1
        fig.savefig("%s/%d_%d.png" % (sample_path, epoch,batch_i),dpi=500,bbox_inches = 'tight')
        plt.close()
        
        
        
        
        
  