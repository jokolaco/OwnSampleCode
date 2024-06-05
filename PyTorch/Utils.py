import torch
import torch.nn as nn
import torch.nn.functional as F
import os           ## for autostop
import shutil
import datetime


####
def save_model(model, optimizer, epoch, iters, savename):
    """ Saving model checkpoint """
    
    if(not os.path.exists("models")):
        os.makedirs("models")
    savepath = f"models/"+savename

    torch.save({
        'iters': iters,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),

    }, savepath)
    return

######

def load_model(model, optimizer, savepath):
    """ Loading pretrained checkpoint """
    
    checkpoint = torch.load(savepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]
    iters = checkpoint["iters"]
    
    return model, optimizer, epoch, iters


###### 

def autostop(time):
    now = datetime.datetime.now()
    hour=now.strftime("%H")
    if (int(hour)) >= time : exit()


