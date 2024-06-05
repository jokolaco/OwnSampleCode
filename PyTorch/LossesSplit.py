import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import kornia
import lpips

########################
########################


class SequenceL1Loss(nn.Module):
    
    def __init__(self):

        super().__init__()
        
        self.loss = nn.L1Loss()

    def forward(self, x, y): 
        
        b_seq_loss = []
        
        b_len, n, c, h, w = x.size()

        x = x.view(b_len * n, c, h, w)
        
        y = y.view(b_len * n, c, h, w)

        current_seq_loss = self.loss(x, y)
        
        b_seq_loss.append(current_seq_loss)

        b_seq_loss = torch.stack(b_seq_loss)   
        
        return b_seq_loss.mean()
        
########################
########################
        
class SequenceSplitL1Loss(nn.Module):


    def __init__(self):
        
        super().__init__()
        
        self.loss = nn.L1Loss()

    def forward(self, x, y):  
        
        b_len = y.shape[0]  ### get batch size
        
        b_seq_lossR = []
        b_seq_lossP = []
        
        for b in range(b_len):
            
            current_seq_loss_Reconstruction = self.loss(x[b,0:10], y[b,0:10])
            current_seq_loss_Prediction = self.loss(x[b,10:20], y[b,10:20])
            
            b_seq_lossR.append(current_seq_loss_Reconstruction)
            b_seq_lossP.append(current_seq_loss_Prediction)
            
        b_seq_lossR = torch.stack(b_seq_lossR)          
        b_seq_lossP = torch.stack(b_seq_lossP)  
        
        return b_seq_lossR.mean(), b_seq_lossP.mean()


########################
########################


class SequenceMSE(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.loss = nn.MSELoss()

    def forward(self, x, y):  
        
      
        b_seq_loss = []
        
        b_len, n, c, h, w = x.size()

        x = x.view(b_len * n, c, h, w)
        
        y = y.view(b_len * n, c, h, w)

        current_seq_loss = self.loss(x, y)
        
        b_seq_loss.append(current_seq_loss)

        b_seq_loss = torch.stack(b_seq_loss)  
        
        return b_seq_loss.mean()



########################
########################


class SequenceSplitMSE(nn.Module):


    def __init__(self):
        
        super().__init__()
        
        self.loss = nn.MSELoss()

    def forward(self, x, y): 
        
        b_len = y.shape[0]  
        
        b_seq_lossR = []
        b_seq_lossP = []
        
        for b in range(b_len):
            
            current_seq_loss_Reconstruction = self.loss(x[b,0:10], y[b,0:10])
            current_seq_loss_Prediction = self.loss(x[b,10:20], y[b,10:20])
            
            b_seq_lossR.append(current_seq_loss_Reconstruction)
            b_seq_lossP.append(current_seq_loss_Prediction)
            
        b_seq_lossR = torch.stack(b_seq_lossR)          
        b_seq_lossP = torch.stack(b_seq_lossP)  
        
        return b_seq_lossR.mean(), b_seq_lossP.mean()

########################
########################

class SequenceSSIM_Loss(nn.Module):
    
    def __init__(self):
        
        super().__init__()

    def forward(self, x, y): 
        
        b_len, n, c, h, w = x.size()

        x = x.view(b_len * n, c, h, w)
        
        y = y.view(b_len * n, c, h, w)
        
        b_seq_loss = []

        current_seq_loss = kornia.losses.ssim_loss(x, y,5)
        
        b_seq_loss.append(current_seq_loss)

        b_seq_loss = torch.stack(b_seq_loss)  
        
        return b_seq_loss.mean()
        
        
        
########################
########################


class SequenceSplitSSIM_Loss(nn.Module):


    def __init__(self):
        
        super().__init__()

    def forward(self, x, y): 
        
        b_len = y.shape[0]  
        
        b_seq_lossR = []
        b_seq_lossP = []
        
        for b in range(b_len):
            
            current_seq_loss_Reconstruction = kornia.losses.ssim_loss(x[b,0:10], y[b,0:10],5)
            current_seq_loss_Prediction = kornia.losses.ssim_loss(x[b,10:20], y[b,10:20],5)
            
            b_seq_lossR.append(current_seq_loss_Reconstruction)
            b_seq_lossP.append(current_seq_loss_Prediction)
            
        b_seq_lossR = torch.stack(b_seq_lossR)          
        b_seq_lossP = torch.stack(b_seq_lossP)  
        
        return b_seq_lossR.mean(), b_seq_lossP.mean()



########################
########################

class SequencePSNR_Loss(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
    def forward(self, x, y):   
        
        b_len, n, c, h, w = x.size()

        x = x.view(b_len * n, c, h, w)
        
        y = y.view(b_len * n, c, h, w)
        
        b_seq_loss = []
 
        current_seq_loss = kornia.losses.psnr_loss(x, y,.5)
        
        b_seq_loss.append(current_seq_loss)
 
        b_seq_loss = torch.stack(b_seq_loss)  
        
        return b_seq_loss.mean()

########################
########################


class SequenceSplitPSNR_Loss(nn.Module):


    def __init__(self):
        
        super().__init__()
        
    def forward(self, x, y):
        
        b_len = y.shape[0]
        
        b_seq_lossR = []
        b_seq_lossP = []
        
        for b in range(b_len):
            
            current_seq_loss_Reconstruction = kornia.losses.psnr_loss(x[b,0:10], y[b,0:10],.5)
            current_seq_loss_Prediction = kornia.losses.psnr_loss(x[b,10:20], y[b,10:20],.5)
            
            b_seq_lossR.append(current_seq_loss_Reconstruction)
            b_seq_lossP.append(current_seq_loss_Prediction)
            
        b_seq_lossR = torch.stack(b_seq_lossR)          
        b_seq_lossP = torch.stack(b_seq_lossP)  
        
        return b_seq_lossR.mean(), b_seq_lossP.mean()



########################
########################

class SequenceLPIPS_Loss(nn.Module):
    
    def __init__(self,network):
        
        super().__init__()
        
        #0/vgg = best forward scores ; 1/alex = closer to "traditional" perceptual loss, when used for optimization
        self.loss = lpips.LPIPS(net='vgg') if (network) else lpips.LPIPS(net='alex')
            

    def forward(self, x, y):  ### x = input ; y = target
        
        b_len, n, c, h, w = x.size()

        x = x.view(b_len * n, c, h, w)
        
        y = y.view(b_len * n, c, h, w)
        
        b_seq_loss = []

        current_seq_loss = self.loss(x, y)
        
        b_seq_loss.append(current_seq_loss)
  
        b_seq_loss = torch.stack(b_seq_loss)  
        
        return b_seq_loss.mean()


########################
########################


class SequenceSplitLPIPS_Loss(nn.Module):


    def __init__(self,network):
        
        super().__init__()
        
        #0/vgg = best forward scores ; 1/alex = closer to "traditional" perceptual loss, when used for optimization
        self.loss = lpips.LPIPS(net='vgg') if (network) else lpips.LPIPS(net='alex')

    def forward(self, x, y):  ### x = input ; y = target
        
        b_len = y.shape[0]  ### get batch size
        
        b_seq_lossR = []
        b_seq_lossP = []
        
        for b in range(b_len):
            
            current_seq_loss_Reconstruction = self.loss(x[b,0:10], y[b,0:10])
            current_seq_loss_Prediction = self.loss(x[b,10:20], y[b,10:20])
            
            b_seq_lossR.append(current_seq_loss_Reconstruction)
            b_seq_lossP.append(current_seq_loss_Prediction)
            
        b_seq_lossR = torch.stack(b_seq_lossR)          
        b_seq_lossP = torch.stack(b_seq_lossP)  
        
        return b_seq_lossR.mean(), b_seq_lossP.mean()




