import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import kornia


#### if we load functions via file we need to specifiy the device here
device="cuda" 

activation=nn.LeakyReLU(0.2, inplace=True) ##nn.GELU()  ### need more RAM.


###################
###################


class HadamardProduct(nn.Module):
    def __init__(self, shape):
        super(HadamardProduct, self).__init__()
        self.weights = nn.Parameter(torch.rand(shape),requires_grad = True)
        
    def forward(self, x):
        return x*self.weights


###################
###################

class PeepConvLSTMCell(nn.Module):

    """Peephole Conv LSTM Implementation p.4 (3) from https://arxiv.org/pdf/1506.04214v1.pdf
        see also in: 'An Optimized Abstractive Text Summarization Model Using Peephole Convolutional LSTM'
        https://pdfs.semanticscholar.org/047a/1e3c0119d5b80b9c8ec1c5b4f6a5f05eebc9.pdf p.6 Figure.2     """

    
    def __init__(self,  in_size, in_channels, hidden_channels ):  ### hidden_shape := length of quadratic canvas aka image size
    
        super(PeepConvLSTMCell, self).__init__()

        xh_channels = in_channels + hidden_channels

        self.in_channels     = in_channels
        self.hidden_channels = hidden_channels
        self.height = in_size[0]   ### to keep the inner states aligned with the input we must set the needed size for the c,h,w states
        self.width  = in_size[1]

        self.W_ci = HadamardProduct((1, hidden_channels, self.height, self.width)) ### HadamardProduct([sizeInput:1,c,w,h]), matrix init happens in functions
        self.W_cf = HadamardProduct((1, hidden_channels, self.height, self.width))
        self.W_co = HadamardProduct((1, hidden_channels, self.height, self.width))

        self.conv_i = nn.Conv2d(xh_channels, xh_channels , kernel_size=3, stride=1, padding=1)  ### possible to change channels intern, but needs adaption
        self.conv_f = nn.Conv2d(xh_channels, xh_channels , kernel_size=3, stride=1, padding=1)
        self.conv_o = nn.Conv2d(xh_channels, xh_channels , kernel_size=3, stride=1, padding=1)
        self.conv_tempc = nn.Conv2d(xh_channels, xh_channels , kernel_size=3, stride=1, padding=1)  ### tempc := temp. cellstate


    def forward(self, x ,  old_hiddenState, old_cellState): 

        xh =  torch.cat([x, old_hiddenState], dim=1)  ### concatenate channels of input x and hidden state -> assumes: image size is equal

        i_x , i_h = torch.split(self.conv_i(xh), self.in_channels, dim=1) ## x= x_t ; h=h_t-1
        f_x , f_h = torch.split(self.conv_f(xh), self.in_channels, dim=1) 
        o_x , o_h = torch.split(self.conv_o(xh), self.in_channels, dim=1) 

        tempc1 , tempc2 = torch.split(self.conv_tempc(xh), self.in_channels, dim=1) 


        i = torch.sigmoid ( i_x + i_h + self.W_ci(old_cellState) )
        f = torch.sigmoid ( f_x + f_h + self.W_cf(old_cellState) )      

        new_cellState = f * old_cellState + i * torch.tanh(tempc1+tempc2)  ### * is the hadamad product -> elementwise multiplication of two matrices

        o = torch.sigmoid ( o_x + o_h + self.W_co(new_cellState) )      

        new_hiddenState = o * torch.tanh(new_cellState)


        return new_hiddenState , new_cellState


    def init_hidden_states(self, batch_size):

        h = torch.randn(batch_size, self.hidden_channels, self.height, self.width,device=device) 
        c = torch.randn(batch_size, self.hidden_channels, self.height, self.width,device=device)
       
        return h, c

###################
###################

class LayeredPeepConvLSTM(nn.Module):   ### toDO: statefulness

    
    def __init__(self, in_size, in_channels, hidden_channels, num_layers):

        super(LayeredPeepConvLSTM, self).__init__()

        self.num_layers = num_layers        

        cell_list = []

        for i in range(0, self.num_layers):

            cell_list.append( PeepConvLSTMCell(in_size, in_channels,hidden_channels) )

        self.cell_list = nn.ModuleList(cell_list) ## modules it contains are properly registered, and will be visible by all Module methods



    def forward(self, inputTensor):   ## inputTensor := x

        b, seq_len, _, h, w = inputTensor.size()

        hidden_states = self._init_hidden_states(batch_size=b)  ### creates array -> hidden_states = []

        cur_layer_input = inputTensor


        for layer_index in range(self.num_layers):

            h, c = hidden_states[layer_index]  

            output_inner = []


            for t in range(seq_len): 

                h, c = self.cell_list[layer_index](x=cur_layer_input[:, t, :, :, :],old_hiddenState = h, old_cellState = c)

                output_inner.append(h)   ### catch output, become intput for next lstm cell / layer


            layer_output = torch.stack(output_inner, dim=1)  
            
            cur_layer_input = layer_output

        return layer_output


    def _init_hidden_states(self, batch_size):

        init_states = []

        for i in range(self.num_layers):

            init_states.append(self.cell_list[i].init_hidden_states(batch_size))

        return init_states


######################## can be used for encoder and decoder, nothing special happens
######################## output: in_channels = out_channels , imageSize constant/no change

class SequenceResConvBlock(nn.Module):
    
    def __init__(self,  in_channels, out_channels, k1 ):  ### with k1 we enlarge the receptive field to grasp more motion (e.g. faster one)
    
        super(SequenceResConvBlock, self).__init__()
        
        if (k1 == 3) : p1=1
        if (k1 == 5) : p1=2
        if (k1 == 7) : p1=3
        if (k1 == 9) : p1=4        


        self.out_channels=out_channels
        self.conv1 = nn.Conv2d(out_channels, out_channels , kernel_size=k1, stride=1, padding=p1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels , kernel_size=k1, stride=1, padding=p1)  
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = activation

    
    def forward(self, x):
    
        b_len, n, c, h, w = x.size()

        x = x.view(b_len * n, c, h, w)

        identity = x

        g = self.conv1(x) 
        g = self.bn1(g)
        g = self.relu(g)

        g = self.conv2(g)  
        g = self.bn2(g)

        g = g + identity   

        y = self.relu(g)   

        y= y.view(b_len , n, c , h, w)

        return y

######################## 
######################## 

class SequenceResConvNeXtBlock(nn.Module): 
    
    def __init__(self,  in_channels, out_channels, k1 ):  ### with we enlarge the receptive field to grasp more motion (e.g. faster one)
    
        super(SequenceResConvNeXtBlock, self).__init__()
        
        if (k1 == 3) : p1=1
        if (k1 == 5) : p1=2
        if (k1 == 7) : p1=3
        if (k1 == 9) : p1=4        
        if (k1 == 11) : p1=5        

 
        self.conv1 = nn.Conv2d(out_channels, out_channels , kernel_size=k1, stride=1, padding=p1,groups=out_channels)  ### depthwise conv -> group conv with #groups = #channels
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, 4*out_channels , kernel_size=1, stride=1, padding=0)  

        self.conv3 = nn.Conv2d(4*out_channels, out_channels , kernel_size=1, stride=1, padding=0)  
        self.gelu = nn.GELU() 

    
    def forward(self, x):
    
        b_len, n, c, h, w = x.size()

        x = x.view(b_len * n, c, h, w)

        identity = x

        g = self.conv1(x) 
        #g = self.bn1(g)

        g = F.layer_norm(g,[c,h,w])

        g = self.conv2(g)  
 
        g = self.gelu(g) 

        g = self.conv3(g) 

        y = g + identity   

        y= y.view(b_len , n, c , h, w)
 
        return y



########################
######################## S E Q U E N C E   D O W N S C A L E   B L O C K


class PixelShuffle_2_Downsampling(nn.Module):

    def __init__(self, in_channels, out_channels):
    
        super(PixelShuffle_2_Downsampling, self).__init__()
        
        self.pixelunshuffle = nn.PixelUnshuffle(2)   ### imagesize/2 , channels * 4

        self.channelTransform = nn.Conv2d(in_channels*4, out_channels , kernel_size=3, stride=1, padding=1)  ### adjust channels to pipeline


    def forward(self, x):

        x = self.pixelunshuffle(x)
        x = self.channelTransform(x)

        return x

########################
########################

class Sequence_EncoderPixelShuffle_2_DownscaleBlock(nn.Module):
    
    def __init__(self,  in_channels, out_channels):
    
        super(Sequence_EncoderPixelShuffle_2_DownscaleBlock, self).__init__()

        self.out_channels=out_channels
        
        self.pixelunshuffle = PixelShuffle_2_Downsampling(in_channels, out_channels)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels , kernel_size=3, stride=1, padding=1)  
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = activation
        
 

    def forward(self, x):

        b_len, n, c, h, w = x.size()
        
        ### since the interpreter don´t know the exact shape of the whole result after b iterations we must initialize the whole array before
        ### maybe there is some .view or .as_strided option to make it faster

        y=torch.empty( (b_len,n,self.out_channels,int(h/2),int(w/2) ),device=device )  

        for b in range(b_len):
    
            identity = x[b]                   ### x = [batch,frames,c,h,w ]
            g = self.pixelunshuffle(x[b])     ### normal: [C,x,y] -> [C/4,x*2,y*2]  for r=2, min. C=4 ; pixelunshuffle: [C,x,y] -> [a,x*2,y*2] ; a=argument user provided
            g = self.bn1(g)
            g = self.relu(g)
            
            g = self.conv2(g)  
            g = self.bn2(g)
            
            identity = self.pixelunshuffle(identity)
     
            g  = g + identity   
            y[b] = self.relu(g)  
        
        return y

######################## 
########################   S E Q U E N C E   U P S C A L E   B L O C K
######################## 



class PixelShuffle_2_Upsampling(nn.Module):

    def __init__(self, in_channels, out_channels):
    
        super(PixelShuffle_2_Upsampling, self).__init__()
        
        self.pixelshuffle = nn.PixelShuffle(2)   ### unpack in space

        self.channelTransform = nn.Conv2d(int(in_channels/4), out_channels , kernel_size=3, stride=1, padding=1)  ### adjust channels to pipeline


    def forward(self, x):

        x = self.pixelshuffle(x)
        x = self.channelTransform(x)

        return x


########################
########################


class Sequence_DecoderPixelShuffle_2_UpscaleBlock(nn.Module):
    
    def __init__(self,  in_channels, out_channels):
    
        super(Sequence_DecoderPixelShuffle_2_UpscaleBlock, self).__init__()

        self.out_channels=out_channels
        
        self.pixelshuffle = PixelShuffle_2_Upsampling(in_channels, out_channels)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels , kernel_size=3, stride=1, padding=1)  
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = activation
        
 

    def forward(self, x):

        b_len, n, c, h, w = x.size()
        
        y=torch.empty( (b_len,n,self.out_channels,h*2,w*2),device=device )   ### since the interpreter don´t know the exact shape of the whole result after b iterations we must initialize the whole array before

        for b in range(b_len):
    
            identity = x[b]                   ### x = [batch,frames,c,h,w ]
            g = self.pixelshuffle(x[b])       ### normal: [C,x,y] -> [C/4,x*2,y*2]  for r=2, min. C=4 ; pixelshuffle: [C,x,y] -> [a,x*2,y*2] ; a=argument user provided
            g = self.bn1(g)
            g = self.relu(g)
            
            g = self.conv2(g)  
            g = self.bn2(g)
            

            identity = self.pixelshuffle(identity)
     
            g = g + identity   
            y[b] = self.relu(g)  
        
        return y


########################
########################


class LSTM_Decoder_Merge(nn.Module):
    
    '''Merge Function Implemented as described in Video Ladder Networks p.3 (6) : https://arxiv.org/abs/1612.01756
        but used a GELU instead of LeakyReLU'''


    def __init__(self,  in_channels):           
    
        super(LSTM_Decoder_Merge, self).__init__()

        self.conv1 = nn.Conv2d(in_channels * 2, in_channels , kernel_size=3, stride=1, padding=1)  
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels , kernel_size=3, stride=1, padding=1)  
        
        ### if there is no input from upper decoder layer (e.g. at the bottom) we use this conv with reduced channels
        self.conv3 = nn.Conv2d(in_channels, in_channels , kernel_size=3, stride=1, padding=1)  
                
        self.relu = nn.GELU()

    def forward(self, x_enc ,h_lstm ,x_dec ):       ## h_lstm = lstm_output

        if (x_dec != None):
            doublecat =  torch.cat([x_dec, h_lstm], dim=1) 
            convresult1   = self.conv1(doublecat)     
        else:
            convresult1   = self.conv3(h_lstm)   

        doublecat2 =  self.relu( torch.cat([convresult1, x_enc], dim=1) )

        h =  self.relu( self.conv2(doublecat2) )

        return h
    
    
class Sequence_LSTM_Decoder_Merge(nn.Module):
    
    def __init__(self,  in_channels):           
    
        super(Sequence_LSTM_Decoder_Merge, self).__init__()

        self.merge = LSTM_Decoder_Merge(in_channels)
        

    def forward(self, x_enc ,h_lstm ,x_dec ): 
       
        b_len, n, c, h, w = h_lstm.size()
        
        y=torch.empty( (b_len,n,c,h,w),device=device )

        
        for b in range(b_len):

            y[b] = self.merge(x_enc[b] ,h_lstm[b] ,x_dec[b]) if (x_dec != None) else self.merge(x_enc[b] ,h_lstm[b],x_dec=None)

        return y      



########################
########################

class Decoder_Conv_Downscale_Block(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        
        self.AdaptConv = Sequence_EncoderPixelShuffle_2_DownscaleBlock(in_channels,out_channels)
        self.Conv      = SequenceResConvBlock(in_channels,out_channels)


    def forward(self, x):  
        
        x = self.AdaptConv(x)
        x = self.Conv(x)

        return x


########################
########################

class Encoder_DeConv_UPscale_Merge_Block(nn.Module):
    
    def __init__(self,  in_channels, out_channels, k1):
        
        super().__init__()
        
        self.Merge        = Sequence_LSTM_Decoder_Merge(in_channels)
        self.Pixelshuffle = Sequence_DecoderPixelShuffle_2_UpscaleBlock(in_channels,out_channels)
        self.Conv         = SequenceResConvNeXtBlock(out_channels,out_channels,k1)


    def forward(self, x ,h_lstm ,x_dec):  
        
        x = self.Merge(x ,h_lstm ,x_dec)
        x = self.Pixelshuffle(x)
        x = self.Conv(x)
        
        return x 
    

########################
########################  BASIC BLOCK ENCODER (INPUT)

class Decoder_Conv_Downscale_LSTM_Block(nn.Module):
    
    def __init__(self, in_size, in_channels, out_channels, num_lstm_layers, k1):
        
        super().__init__()
        
        h= int(in_size[0]/2)
        w= int(in_size[1]/2)
        
        self.AdaptConv = Sequence_EncoderPixelShuffle_2_DownscaleBlock(in_channels,out_channels)
        self.Conv      = SequenceResConvNeXtBlock(out_channels,out_channels,k1)
        self.Lstm      = LayeredPeepConvLSTM([h,w],out_channels,out_channels,num_lstm_layers)

    def forward(self, x):  
        
        x = self.AdaptConv(x)
        x = self.Conv(x)
        
        y = self.Lstm(x)

        return x , y


########################
