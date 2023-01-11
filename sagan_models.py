import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from spectral import SpectralNorm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale


class Sin_activate(torch.nn.Module):#sin激活函数
    def __init__(self):
        super().__init__()

    def forward(self,x):
        x=x.sin()
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        layer = []
        layer.append(nn.Conv1d(in_features, in_features, 3, padding=[1]))
        layer.append(nn.InstanceNorm1d(in_features))
        layer.append(Sin_activate())
        
        
        layer.append(nn.Conv1d(in_features, in_features, 3, padding=[1]))
        layer.append(nn.InstanceNorm1d(in_features))
        layer.append(Sin_activate())
        
        
        self.conv_block = nn.Sequential(*layer)

    def forward(self, x):
        # return x + self.conv_block(x)
        return self.conv_block(x)


class ResidualBlock9(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        layer = []
        layer.append(nn.Conv1d(in_features, in_features, 3, padding=[1]))
        layer.append(nn.InstanceNorm1d(in_features))
        layer.append(Sin_activate())
        layer.append(nn.Conv1d(in_features, in_features, 3, padding=[1]))
        layer.append(nn.InstanceNorm1d(in_features))
        
        self.conv_block = nn.Sequential(*layer)

    def forward(self, x):
        return x + self.conv_block(x)





class CoAtt(nn.Module):
    def __init__(self,in_dim):
        super(CoAtt,self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation
        
        self.query_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.key_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim, kernel_size= 1)
        self.value_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
        
        
    def forward(self,x):
        # print(x.shape)
        mean, std = torch.mean(x), torch.std(x)
        x_zscore  = (x-mean)/std
        m_batchsize_z,C_z,width_z = x_zscore.size()
        height_z = 1
        x_transpose =x_zscore.view(m_batchsize_z,-1,width_z*height_z).permute(0,2,1)
        
        m_batchsize,C,width = x.size()
        height = 1
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height)      
        query_energy =  torch.bmm(x_transpose,proj_query).permute(0,2,1)
        
        
        
        x_flatten =x_zscore.view(m_batchsize_z,-1,width_z*height_z)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height)
        key_energy =  torch.bmm(x_transpose,proj_key)
        
        
        energy =  torch.bmm(query_energy,key_energy)
        attention = self.softmax(energy)
        
        
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height)     
        out = torch.bmm(proj_value,attention.permute(0,2,1))           
        out = self.gamma*out + x_flatten
        # out = self.gamma*out
        out = out.view(m_batchsize,C,width)  
        # print(out)
  
    
        return out



    
    
class ResidualBlock2(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock2, self).__init__()

        layer = []
        layer.append(nn.Conv1d(in_features, in_features, 3, padding=[1]))
        layer.append(Sin_activate())
        layer.append(nn.InstanceNorm1d(in_features))       
        layer.append(nn.Conv1d(in_features, in_features, 3, padding=[1]))
        layer.append(Sin_activate())
        layer.append(nn.InstanceNorm1d(in_features))
        layer.append(CoAtt(in_features))
        
        
        self.conv_block = nn.Sequential(*layer)

    def forward(self, x):
        return x + self.conv_block(x)



class CCAR(nn.Module):
    def __init__(self,in_dim):
        super(CCAR,self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation
        self.query_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.key_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim, kernel_size= 1)
        self.value_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.residual = ResidualBlock(in_dim)
        self.softmax  = nn.Softmax(dim=-1) #
        
        
    def forward(self,x):
        
        g = self.residual(x)
        x_g = x + g
        
        # print(x.shape)
        mean, std = torch.mean(x_g), torch.std(x_g)
        x_g_zscore  = (x_g-mean)/std
        m_batchsize_x_g, C_x_g, width_x_g = x_g_zscore.size()
        height_x_g = 1
        x_g_transpose =x_g_zscore.view(m_batchsize_x_g,-1,width_x_g*height_x_g).permute(0,2,1)
        
        m_batchsize,C,width = x_g.size()
        height = 1
        proj_query  = self.query_conv(x_g).view(m_batchsize,-1,width*height)      
        query_energy =  torch.bmm(x_g_transpose,proj_query).permute(0,2,1)
        
        
        
        mean, std = torch.mean(g), torch.std(g)
        g_zscore  = (g-mean)/std
        m_batchsize_g, C_g, width_g = x_g_zscore.size()
        height_g = 1
        
        
        g_flatten =g_zscore.view(m_batchsize_g,-1,width_g*height_g)
        g_transpose = g_flatten.permute(0,2,1)
        
        proj_key =  self.key_conv(g).view(m_batchsize_g,-1,width_g*height_g)
        key_energy =  torch.bmm(g_transpose,proj_key)
        energy =  torch.bmm(query_energy,key_energy)
        attention = self.softmax(energy)
        
        
        proj_value = self.value_conv(g).view(m_batchsize_g,-1,width_g*height_g)     
        out = torch.bmm(proj_value,attention.permute(0,2,1))           
        # out = self.gamma*out 
        # out = self.gamma*out + g_flatten
        out = out.view(m_batchsize,C,width)  
        
        # #consider x+g as query
        # proj_value = self.value_conv(x_g).view(m_batchsize_x_g,-1,width_x_g*height_x_g)     
        # out = torch.bmm(proj_value,attention.permute(0,2,1))           
        # out = self.gamma*out
        # # out = self.gamma*out + g_flatten
        # out = out.view(m_batchsize,C,width)
  
    
        return out




class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(rate)
        

    def forward(self, inputs):
        attn_output, attn_output_weights = self.att(inputs,inputs ,inputs)
        attn_output = self.dropout(attn_output)
        out = self.layernorm(inputs * attn_output)

        return out



def multiply(x):
        mask,image  = x
        return image* torch.clamp(mask,0.8,1)


class Generator(nn.Module):
    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64):
        super(Generator, self).__init__()
        self.imsize = image_size
        
        self.transformer  = TransformerBlock(128,2)
        
        #v0
        # self.conv1DWithSINE0 = nn.Sequential(nn.Conv1d(1,1,31, padding=[15]),nn.InstanceNorm1d(1), Sin_activate())
        # self.conv1DWithSINE1 = nn.Sequential(nn.Conv1d(1,13,3, padding=[1]),nn.InstanceNorm1d(13), Sin_activate())       
        # self.conv1DWithSINE2 = nn.Sequential(nn.Conv1d(13,7,5, padding=[2]),nn.InstanceNorm1d(7), Sin_activate())      
        # self.conv1DWithSINE3 = nn.Sequential(nn.Conv1d(7,5,13, padding=[6]),nn.InstanceNorm1d(5), Sin_activate())
        # # self.conv1DWithSINE4 = nn.Sequential(nn.Conv1d(5,1,3, padding=[1]),nn.InstanceNorm1d(1), Sin_activate())
        # self.conv1DWithSINE4 = nn.Sequential(nn.Conv1d(5,1,3, padding=[1]))
        
        
        # v1
        # self.conv1DWithSINE_l0 = nn.Sequential(nn.Conv1d(1,1,31, padding=[15]),nn.InstanceNorm1d(1), Sin_activate())
        # self.conv1DWithSINE_l1 = nn.Sequential(nn.Conv1d(1,5,3, padding=[1]),nn.InstanceNorm1d(5), Sin_activate())
        # self.conv1DWithSINE_l2 = nn.Sequential(nn.Conv1d(5,7,13, padding=[6]),nn.InstanceNorm1d(7), Sin_activate())      
        # self.conv1DWithSINE_l3 = nn.Sequential(nn.Conv1d(7,13,5, padding=[2]),nn.InstanceNorm1d(13), Sin_activate())
            
        # self.conv1DWithSINE_r3 = nn.Sequential(nn.Conv1d(13,7,5, padding=[2]),nn.InstanceNorm1d(7), Sin_activate())      
        # self.conv1DWithSINE_r2 = nn.Sequential(nn.Conv1d(7,5,13, padding=[6]),nn.InstanceNorm1d(5), Sin_activate())
        # self.conv1DWithSINE_r1 = nn.Sequential(nn.Conv1d(5,1,3, padding=[1]))
        
        
        # # v1+ACAE
        # self.conv1DWithSINE_l0 = nn.Sequential(nn.Conv1d(1,1,31, padding=[15]),nn.InstanceNorm1d(1), Sin_activate())
        # self.conv1DWithSINE_l1 = nn.Sequential(nn.Conv1d(1,5,3, padding=[1]),nn.InstanceNorm1d(5), Sin_activate())
        # self.conv1DWithSINE_l2 = nn.Sequential(nn.Conv1d(5,7,13, padding=[6]),nn.InstanceNorm1d(7), Sin_activate())      
        # self.conv1DWithSINE_l3 = nn.Sequential(nn.Conv1d(7,13,5, padding=[2]),nn.InstanceNorm1d(13), Sin_activate(),CoAtt(13))
            
        # self.conv1DWithSINE_r3 = nn.Sequential(nn.Conv1d(13,7,5, padding=[2]),nn.InstanceNorm1d(7), Sin_activate())      
        # self.conv1DWithSINE_r2 = nn.Sequential(nn.Conv1d(7,5,13, padding=[6]),nn.InstanceNorm1d(5), Sin_activate())
        # self.conv1DWithSINE_r1 = nn.Sequential(nn.Conv1d(5,1,3, padding=[1]))
        
        
        
        # v1+ACAE+CCAR
        self.conv1DWithSINE_l0 = nn.Sequential(nn.Conv1d(1,1,31, padding=[15]),nn.InstanceNorm1d(1), Sin_activate())
        self.conv1DWithSINE_l1 = nn.Sequential(nn.Conv1d(1,5,3, padding=[1]),nn.InstanceNorm1d(5), Sin_activate())
        self.conv1DWithSINE_l2 = nn.Sequential(nn.Conv1d(5,7,13, padding=[6]),nn.InstanceNorm1d(7), Sin_activate())      
        self.conv1DWithSINE_l3 = nn.Sequential(nn.Conv1d(7,13,5, padding=[2]),nn.InstanceNorm1d(13), Sin_activate(),CoAtt(13))
            
        self.conv1DWithSINE_r3 = nn.Sequential(nn.Conv1d(13,7,5, padding=[2]),nn.InstanceNorm1d(7), Sin_activate())      
        self.conv1DWithSINE_r2 = nn.Sequential(nn.Conv1d(7,5,13, padding=[6]),nn.InstanceNorm1d(5), Sin_activate())
        self.conv1DWithSINE_r1 = nn.Sequential(nn.Conv1d(5,1,3, padding=[1]))
    
        self.ccar1 = CCAR(5)
        self.ccar2 = CCAR(7)
        self.ccar3 = CCAR(13)

        


    def forward(self, z):
        # baseline
        # value = self.conv1DWithSINE_l0(z.to(torch.float32))
        # att = self.transformer(value)
        # mean, std = torch.mean(att), torch.std(att)
        # att  = (att-mean)/std  
        # l00 = value* torch.clamp(att,0.8,1)
        

        # l01 = self.conv1DWithSINE_l1(l00)   
        # l02 = self.conv1DWithSINE_l2(l01)
        # l03 = self.conv1DWithSINE_l3(l02) 
    

        
        # r03 = self.conv1DWithSINE_r3(l03)   
        # r02 = self.conv1DWithSINE_r2(r03)
        # r01 = self.conv1DWithSINE_r1(r02) 
        
        
        
        value = self.conv1DWithSINE_l0(z.to(torch.float32))
        att = self.transformer(value)
        mean, std = torch.mean(att), torch.std(att)
        att  = (att-mean)/std  
        l00 = value* torch.clamp(att,0.8,1)
        

        l01 = self.conv1DWithSINE_l1(l00)   
        l02 = self.conv1DWithSINE_l2(l01)
        l03 = self.conv1DWithSINE_l3(l02) 
        
        
        l03_ccar = self.ccar3(l03)
        l02_ccar = self.ccar2(l02)
        l01_ccar = self.ccar1(l01)

        
        
        
        l03 = l03 + l03_ccar
        # l03 = l03
        r03 = self.conv1DWithSINE_r3(l03)  
        
        r03 = r03 + l02_ccar
        # r03 = r03 + l02
        r02 = self.conv1DWithSINE_r2(r03)
        
        r02 = r02 + l01_ccar 
        # r02 = r02 + l01   
        r01 = self.conv1DWithSINE_r1(r02) 
        
        
        return r01

    


class Discriminator(nn.Module):
    def __init__(self, batch_size=64, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
        
        self.conv1DWithSINE0 = nn.Sequential(nn.Conv1d(1,12,5, padding=[2]),nn.InstanceNorm1d(12), Sin_activate())
        self.conv1DWithSINE1 = nn.Sequential(nn.Conv1d(12,7,5, padding=[2]),nn.InstanceNorm1d(7), Sin_activate())       
        self.conv1DWithSINE2 = nn.Sequential(nn.Conv1d(7,3,5, padding=[2]),nn.InstanceNorm1d(3), Sin_activate())      
        self.conv1DWithSINE3 = nn.Sequential(nn.Conv1d(3,1,5, padding=[2]),nn.InstanceNorm1d(1), Sin_activate())


        self.attn1 = CoAtt(256)
        self.attn2 = CoAtt(512)
        self.attn3 = CoAtt(1024)

    def forward(self, x):
        # print(x.shape)
        out = self.conv1DWithSINE0(x.to(torch.float32))
        # print(out.shape)
        out = self.conv1DWithSINE1(out)
        # print(out.shape)
        out = self.conv1DWithSINE2(out)
        # print(out.shape)
        out = self.conv1DWithSINE3(out)
        # print(out.shape)


        return out
