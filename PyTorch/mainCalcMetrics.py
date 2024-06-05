import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy,DALIGenericIterator

from Losses import *
from LossesSplit import *

from NetFunctions import *
from torchinfo import summary

from NetFunctions import *
from Utils import *
from CustomSaveImage import *

from torch.utils.tensorboard import SummaryWriter
import kornia

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


## Dali Dataloader
stopper=0
max_batch_size =8
sequence_length = 20

h=   64 
w=   64 

normalize=True


train_dir = "/data/KTH_Sequence/train"
eval_dir  = "/data/KTH_Sequence/eval"

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


### Since we deviate from the regular batch scheme, where each element of the batch is one frame,
### we cannot use dataloader statistics/feedback and must calculate our own 

epochsTrain = math.ceil(pipeTrain.epoch_size("Reader")/max_batch_size)
epochsEval = math.ceil(pipeEval.epoch_size("Reader")/max_batch_size)
print(epochsTrain,epochsEval)


# **TensorBoard**

# TensorBoard / Imageoutput Directories


deleteOldImages=True

TBOARD_LOGS = os.path.join(os.getcwd(), "tboard_logs", "VLN_Net_Metrics")

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
                                                        ### Imagesize, inchan. , outchan., layer_num ,k1
        self.decBlock1 = Decoder_Conv_Downscale_LSTM_Block(inputSize        ,3   ,64   ,4,9)  
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


model = VLN_Net().to(device)


#### Training / Evaluation Functions

#####################
#####################  File Output

OutputFile_L1=open("statsL1.dat",'w')
OutputFile_L2=open("statsL2.dat",'w')
OutputFile_SSIM=open("statsSSIM.dat",'w')
OutputFile_PSNR=open("statsPSNR.dat",'w')
OutputFile_LPIPS=open("statsLPIPS.dat",'w')
OutputFile_MIXED=open("statsMIXED.dat",'w')
OutputFile_ModelInfo=open("ModelInfo.txt",'w')

@torch.no_grad()
def CalcMetricsModel(model, eval_loader, device):
    """ Evaluating the model for either validation or test """
    criterion1 = SequenceL1Loss().to(device)
    criterion1S = SequenceSplitL1Loss().to(device)

    criterion2 = SequenceMSE().to(device)
    criterion2S = SequenceSplitMSE().to(device)

    criterion3 = SequenceSSIM_Loss().to(device)
    criterion3S = SequenceSplitSSIM_Loss().to(device)
    
    criterion4 = SequencePSNR_Loss().to(device)
    criterion4S = SequenceSplitPSNR_Loss().to(device)
    
    criterion5 = SequenceLPIPS_Loss(0).to(device)
    criterion5S = SequenceSplitLPIPS_Loss(0).to(device)
  
    iterationCounterEval = 0
    
    print(" Metrics Evaluation")

    for i in range(epochsEval):
        
        data= next(iter(Dali_GenericEvaluationLoader)) 

        OrginalImages = data[0]["Plain"].to(device)

        UncompleteImages= OrginalImages.detach().clone()   ### clone needs some time but its ok

        UncompleteImages[:,10:20]=0   ### delete frames
        outputs = model(UncompleteImages)
        
        loss1=criterion1(OrginalImages,outputs)
        loss1R,loss1P=criterion1S(OrginalImages,outputs)       
        
        loss2=criterion2(OrginalImages,outputs)
        loss2R,loss2P=criterion2S(OrginalImages,outputs)

        loss3=criterion3(OrginalImages,outputs)
        loss3R,loss3P=criterion3S(OrginalImages,outputs)

        loss4=criterion4(OrginalImages,outputs)
        loss4R,loss4P=criterion4S(OrginalImages,outputs)       

        loss5=criterion5(OrginalImages,outputs)
        loss5R,loss5P=criterion5S(OrginalImages,outputs)  
        
        
        #### Tensorboard Dashboard preview and instant feedback

        writer.add_scalar(f'Loss/VLN Metrics L1/MAE ', loss1.item(), global_step=iterationCounterEval)
        writer.add_scalar(f'Loss/VLN Metrics L2/MSE ', loss2.item(), global_step=iterationCounterEval)
        writer.add_scalar(f'Loss/VLN Metrics SSIM   ', loss3.item(), global_step=iterationCounterEval)
        writer.add_scalar(f'Loss/VLN Metrics PSNR   ', loss4.item(), global_step=iterationCounterEval)
        writer.add_scalar(f'Loss/VLN Metrics LPIPS  ', loss5.item(), global_step=iterationCounterEval)
     
        writer.add_scalars(f'Metrics/MSE', {
                            'Prediction ': loss2P,
                            'Reconstruction ': loss2R,
                        }, iterationCounterEval)  


        writer.add_scalars(f'Metrics/L1', {
                            'Prediction ': loss1P,
                            'Reconstruction ': loss1R,
                        }, iterationCounterEval)  
                        
                        
        writer.add_scalars(f'Metrics/SSIM', {
                            'Prediction ': loss3P,
                            'Reconstruction ': loss3R,
                        }, iterationCounterEval)  
                        
        writer.add_scalars(f'Metrics/PSNR', {
                            'Prediction ': loss4P,
                            'Reconstruction ': loss4R,
                        }, iterationCounterEval)   
                        
                        
        writer.add_scalars(f'Metrics/LPIPS', {
                            'Prediction ': loss5P,
                            'Reconstruction ': loss5R,
                        }, iterationCounterEval)   

        #### File Output for further data analysis
        
        OutputFile_L1.write(str(iterationCounterEval)+" ")
        OutputFile_L1.write(str(loss1P.item())+" ")
        OutputFile_L1.write(str(loss1R.item())+" ")
        OutputFile_L1.write(str(loss1R.item()-loss1P.item())+" ")
        OutputFile_L1.write(str(loss1R.item()+loss1P.item())+" ")
        OutputFile_L1.write("\n")

        OutputFile_L2.write(str(iterationCounterEval)+" ")
        OutputFile_L2.write(str(loss2P.item())+" ")
        OutputFile_L2.write(str(loss2R.item())+" ")
        OutputFile_L2.write(str(loss2R.item()-loss2P.item())+" ")
        OutputFile_L2.write(str(loss2R.item()+loss2P.item())+" ")
        OutputFile_L2.write("\n")
        
        OutputFile_SSIM.write(str(iterationCounterEval)+" ")
        OutputFile_SSIM.write(str(loss3P.item())+" ")
        OutputFile_SSIM.write(str(loss3R.item())+" ")
        OutputFile_SSIM.write(str(loss3R.item()-loss3P.item())+" ")
        OutputFile_SSIM.write(str(loss3R.item()+loss3P.item())+" ")
        OutputFile_SSIM.write("\n")
        
        OutputFile_PSNR.write(str(iterationCounterEval)+" ")
        OutputFile_PSNR.write(str(loss4P.item())+" ")
        OutputFile_PSNR.write(str(loss4R.item())+" ")
        OutputFile_PSNR.write(str(loss4R.item()-loss4P.item())+" ")
        OutputFile_PSNR.write(str(loss4R.item()+loss4P.item())+" ")
        OutputFile_PSNR.write("\n")
        
        OutputFile_LPIPS.write(str(iterationCounterEval)+" ")
        OutputFile_LPIPS.write(str(loss5P.item())+" ")
        OutputFile_LPIPS.write(str(loss5R.item())+" ")
        OutputFile_LPIPS.write(str(loss5R.item()-loss5P.item())+" ")
        OutputFile_LPIPS.write(str(loss5R.item()+loss5P.item())+" ")
        OutputFile_LPIPS.write("\n")
                
        OutputFile_MIXED.write(str(iterationCounterEval)+" ")        
        OutputFile_MIXED.write(str(loss1P.item()+loss2P.item()+loss3P.item()+loss4P.item()+loss5P.item())+" ")
        OutputFile_MIXED.write(str(loss1R.item()+loss2R.item()+loss3R.item()+loss4R.item()+loss5R.item())+" ")
        OutputFile_MIXED.write(str(loss1P.item()+loss2P.item()+loss3P.item()+loss4P.item()+loss5P.item()-loss1R.item()-loss2R.item()-loss3R.item()-loss4R.item()-loss5R.item())+" ")
        OutputFile_MIXED.write(str(loss1R.item()+loss2R.item()+loss3R.item()+loss4R.item()+loss5R.item()+loss1R.item()+loss2R.item()+loss3R.item()+loss4R.item()+loss5R.item())+" ")
        OutputFile_MIXED.write("\n")
        

        iterationCounterEval+=1


    OutputFile_L1.close()
    OutputFile_L2.close()
    OutputFile_SSIM.close()
    OutputFile_PSNR.close()
    OutputFile_LPIPS.close()
    OutputFile_MIXED.close()

    return iterationCounterEval



########################
########################
optimizer=optim.AdamW(model.parameters(), lr=3e-4)

load_model(model=model,optimizer=optimizer,savepath="models/VLN_model")

OutputFile_ModelInfo.write(str(summary(model, input_size=(8, 20, 3, 64,64))))  ### ensure that image/batchsize is correct

CalcMetricsModel(model=model, eval_loader=Dali_GenericEvaluationLoader, device=device)
