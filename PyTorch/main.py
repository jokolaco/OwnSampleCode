import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy,DALIGenericIterator

from NetFunctions import *
from Utils import *
from Losses import *
from CustomSaveImage import *

from torch.utils.tensorboard import SummaryWriter
import kornia

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Dali Dataloader

max_batch_size =4
sequence_length = 20

h=   64  
w=   64  

normalize=True


train_dir = "/data/KTH_Sequence_Debug/train"
eval_dir = "/data/KTH_Sequence_Debug/eval"


@pipeline_def
def dali_VideoPipeline(normalize,training):
    
    image_dir = train_dir if (training) else eval_dir 

    video= fn.readers.sequence(device="cpu",file_root=image_dir , sequence_length = sequence_length, name="Reader")
    
    if (normalize) : video = fn.crop_mirror_normalize(video,
                                      output_layout="FCHW",
                                      
                                      mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                                      std=[0.5 * 255, 0.5 * 255, 0.5 * 255]

                                  )
    
    
    #video = fn.resize(video, size=(128,160),mode="not_smaller") ### keeping the aspect ratio, so that no extent of the output image is smaller than specified.
    video = fn.resize(video, size=(h,w))

    return video

pipeTrain =  dali_VideoPipeline(normalize=normalize,training=True, batch_size=max_batch_size, num_threads=8, device_id=0, seed=222)
pipeTrain.build()

pipeEval =  dali_VideoPipeline(normalize=normalize,training=False, batch_size=max_batch_size, num_threads=8, device_id=0, seed=222)
pipeEval.build()


Dali_GenericTrainLoader = DALIGenericIterator([pipeTrain], output_map=["Plain"],auto_reset =True,last_batch_policy=LastBatchPolicy.DROP)

Dali_GenericEvaluationLoader = DALIGenericIterator([pipeEval], output_map=["Plain"],auto_reset =True,last_batch_policy=LastBatchPolicy.DROP)

### Since we deviate from the regular batch scheme, were each element of the batch is one frame,
### we cannot use dataloader statistics/feedback and must calculate our own 

epochsTrain = math.ceil(pipeTrain.epoch_size("Reader")/max_batch_size)
epochsEval = math.ceil(pipeEval.epoch_size("Reader")/max_batch_size)
print(epochsTrain,epochsEval)


# TensorBoard / Imageoutput Directories


deleteOldImages=True

TBOARD_LOGS = os.path.join(os.getcwd(), "tboard_logs", "VLN_Net")

if not os.path.exists(TBOARD_LOGS):
    os.makedirs(TBOARD_LOGS)

    
shutil.rmtree(TBOARD_LOGS)
writer = SummaryWriter(TBOARD_LOGS)


savepath = "imgs/VLN_progress"
if os.path.exists(savepath) and (deleteOldImages): shutil.rmtree(savepath)
if not os.path.exists(savepath):
        os.makedirs(savepath)



### The Model

inputSize = torch.Tensor([h,w]).to(device)

class VLN_Net(nn.Module):
    
    def __init__(self):
    
        super().__init__()

        self.decBlock1 = Decoder_Conv_Downscale_LSTM_Block(inputSize        ,3   ,64   ,4,9)  ### Imagesize, inchan. , outchan., layer_num ,k1
        self.decBlock2 = Decoder_Conv_Downscale_LSTM_Block(inputSize.div(2) ,64  ,96   ,4,9)
        self.decBlock3 = Decoder_Conv_Downscale_LSTM_Block(inputSize.div(4) ,96  ,128  ,4,7)
        self.decBlock4 = Decoder_Conv_Downscale_LSTM_Block(inputSize.div(8) ,128 ,192  ,4,7)

        self.encMergeBlock1 = Encoder_DeConv_UPscale_Merge_Block(192  ,128,9 )
        self.encMergeBlock2 = Encoder_DeConv_UPscale_Merge_Block(128 ,96,9)
        self.encMergeBlock3 = Encoder_DeConv_UPscale_Merge_Block(96  ,64,7)
        self.encMergeBlock4 = Encoder_DeConv_UPscale_Merge_Block(64  ,3,7)


    def forward(self,x):

        x1,h1 = self.decBlock1(x)
        x2,h2 = self.decBlock2(x1)
        x3,h3 = self.decBlock3(x2)
        x4,h4 = self.decBlock4(x3)

        y4 = self.encMergeBlock1(x4,h4,None)
        y3 = self.encMergeBlock2(x3,h3,y4)
        y2 = self.encMergeBlock3(x2,h2,y3)
        y1 = self.encMergeBlock4(x1,h1,y2)

        return y1   

#torch.autograd.set_detect_anomaly(True)
model = VLN_Net().to(device)
criterion = SequenceMSE().to(device)
optimizer=optim.AdamW(model.parameters(), lr=3e-4)
scheduler =optim.lr_scheduler.StepLR(optimizer,  step_size=5, gamma=.01)  ## we don´t use this scheduler often, because one epoch needs at least a few hours. 
                                                                          ## we stop the computation and set the lr per hand.

#### Training / Evaluation Functions

def train_epoch(model, train_loader, optimizer, criterion, epoch, device, ic):

    
    loss_list = []
    
    iterationCounterTrain = ic
    
    print(" Training")
    
    for i in range(epochsTrain):

        feedbackImages = torch.empty(32, 3, h, w)  
        
        data= next(iter(Dali_GenericTrainLoader)) 
              
        OrginalImages = data[0]["Plain"].to(device)

        optimizer.zero_grad()

        UncompleteImages= OrginalImages.detach().clone()   ### clone needs some time but its ok, GPU runs btw. 97-100%

        UncompleteImages[:,10:20]=0   ### delete frames

        outputs = model(UncompleteImages)
        
        loss=criterion(OrginalImages,outputs) ## the indices are right as can be seen from the feedback images, 
                                              ## suggested indexing like (OrginalImages[1:],outputs[:-1]) produces errors
        loss_list.append(loss.item())
                    
        loss.backward()

        optimizer.step()

        print(epoch, i, loss.item())  ## some feedback for the command line ;-)

        writer.add_scalar(f'Loss/VLN Train Iteration Loss', loss.item(), global_step= iterationCounterTrain)

         ## colorize images for feedback
        #outputs[0,0:10,1]=outputs[0,0:10,1]*2.5     ### green
        #outputs[0,10:20,0]=outputs[0,10:20,0]*2.5   #### red
                    
         ### compare ground truth (GT) and corresponding network output given the respective GT
        feedbackImages[0:8]= OrginalImages[0][2:10]
        feedbackImages[8:16]=outputs[0][2:10]
         ### compare ground truth and network output given NO input , only the previous frames
        feedbackImages[16:24]= OrginalImages[0][10:18]
        feedbackImages[24:32]=outputs[0][10:18]


        ## We use an improved custom saver, with some tweaks regarding the quality/imagesize. JPG is faster but the quality is not so good.
        #save_imageJPG(feedbackImages, os.path.join(savepath, f"VLN_predictions{iterationCounterTrain:06}.jpg"), normalize=True )
        save_imagePNG(feedbackImages, os.path.join(savepath, f"VLN_predictions{iterationCounterTrain:06}.png"), normalize=True )

        if (iterationCounterTrain%1024==0 ): save_model(model, optimizer ,epoch=epoch,iters=iterationCounterTrain, savename="VLN_model")        
            
        iterationCounterTrain+=1

                        
    mean_loss = np.mean(loss_list)
    
    return mean_loss, loss_list,iterationCounterTrain



#####################
#####################


@torch.no_grad()
def eval_model(model, eval_loader, criterion, device, epoch,ic):
    """ Evaluating the model for either validation or test """
    
    loss_list = []

    iterationCounterEval = ic
    
    print(" Evaluation")

    for i in range(epochsEval):
        
        data= next(iter(Dali_GenericEvaluationLoader)) 

        OrginalImages = data[0]["Plain"].to(device)

        UncompleteImages= OrginalImages.detach().clone()   ### clone needs some time but its ok

        UncompleteImages[:,10:20]=0   ### delete frames
                    
        outputs = model(UncompleteImages)
        
        loss=criterion(OrginalImages,outputs)
        
        writer.add_scalar(f'Loss/VLN Eval Iteration Loss', loss.item(), global_step=iterationCounterEval)
        
        print(epoch, i, loss.item())
        
        loss_list.append(loss.item())
                
        iterationCounterEval+=1

    mean_loss = np.mean(loss_list)
    
    return mean_loss, iterationCounterEval




########################
########################

def train_model(model, optimizer, scheduler,criterion, train_loader, eval_loader, num_epochs):
    """ Training a model for a given number of epochs"""
    
    train_loss = []
    val_loss =  []
    loss_iters = []
    
    ict=0
    ice=0
    

    for epoch in range(num_epochs):

        #autostop(10)  ### stop calculations after 10:00
        
        # training epoch
        model.train()  ## set mopel in training mode
        mean_lossTrain, cur_loss_iters,ict = train_epoch(model=model, train_loader=eval_loader, optimizer=optimizer, criterion=criterion, epoch=epoch, device=device, ic=ict )
        
        # validation epoch
        model.eval()  
        
        mean_lossEval,ice = eval_model(model=model, eval_loader=eval_loader, criterion=criterion, device=device, epoch=epoch , ic=ice)
        
        train_loss.append(mean_lossTrain)
        val_loss.append(mean_lossEval)
        
        writer.add_scalars(f'Comb_Loss/EpochLosses', {
                            'TrainEpochLoss': mean_lossTrain,
                            'EvalEpochLoss':  mean_lossEval,
                        }, epoch)    

        scheduler.step()
        
    return train_loss, val_loss


### Reload precomputed Model
#load_model(model=model,optimizer=optimizer,savepath="models/VLN_model")


train_model(model=model, optimizer=optimizer, scheduler=scheduler,
                                               criterion=criterion,
                                               train_loader=Dali_GenericTrainLoader, 
                                               eval_loader=Dali_GenericEvaluationLoader, 
                                               num_epochs=250)


