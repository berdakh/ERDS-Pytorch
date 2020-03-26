# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 19:14:57 2019
@author: berdakh.abibullaev
"""

import torch
from torch.autograd import Variable
import torch.nn as nn 
import numpy as np 
 
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#%%        
def compute_conv_dim(dim_size, kernel_size, padding, stride):
    
    out_dim = int((dim_size - kernel_size + 2 * padding) / stride + 1)    
    return out_dim

#%%       
class CNN2D(torch.nn.Module):  
    """ Flexible 2D CNN 
    Example Usage:
        from nu_models import CNN_2DMod
        model = CNN_2DMod(kernel_size = [3, 3, 3, 3] , conv_channels = [1, 8, 16, 32])    
    """
    def __init__(self, input_size, 
                 kernel_size, 
                 conv_channels,
                 dense_size,
                 dropout  ):    
        
        super(CNN2D, self).__init__()          
        self.cconv   = []  
        self.MaxPool = nn.MaxPool2d((1, 2), (1, 2))  
#        self.MaxPool = nn.MaxPool2d(2, 2, padding = 1)
        self.ReLU    = nn.ReLU()
        self.Dropout = nn.Dropout(dropout)        
        self.batchnorm = []        
        
        for jj in conv_channels:
            self.batchnorm.append(nn.BatchNorm2d(jj, eps=0.001, momentum=0.01,
                                                 affine=True, track_running_stats=True).cuda())               
        ii = 0        
        # define CONV layer architecture:    
        for in_channels, out_channels in zip(conv_channels, conv_channels[1:]):                           
            conv_i = torch.nn.Conv2d(in_channels  = in_channels, 
                                     out_channels = out_channels,
                                     kernel_size  = kernel_size[ii],
                                     padding      = (kernel_size[ii][0]//2, 
                                                     kernel_size[ii][1]//2)
                                     )
            
            self.cconv.append(conv_i)                
            self.add_module('CNN_K{}_O{}'.format(kernel_size[ii], out_channels), conv_i)
            ii += 1            
            
        self.flat_fts = self.get_output_dim(input_size, self.cconv)    
        self.fc1 = torch.nn.Linear(self.flat_fts, dense_size)
        self.fc2 = torch.nn.Linear(dense_size, 2)        
        
    def get_output_dim(self, input_size, cconv):        
        with torch.no_grad():
            input = torch.ones(1,*input_size)              
            for conv_i in cconv:                
                input = conv_i(input)
                input = self.MaxPool(input)        
                flatout = int(np.prod(input.size()[1:]))
                print(input.shape)
                print("Flattened output ::", flatout)                
        return flatout 

    def forward(self, input):        
        for jj, conv_i in enumerate(self.cconv):
            conv_i.cuda()
            
            input = conv_i(input)
            input = self.batchnorm[jj+1](input)
            input = self.ReLU(input)        
            input = self.MaxPool(input)    
               
        # flatten the CNN output     
        out = input.view(-1, self.flat_fts) 
        out = self.fc1(out)                       
        out = self.Dropout(out)        
        out = self.fc2(out)      
        return out        

#%% ## _LSTM Model        
class LSTM_Model(torch.nn.Module):
    """
    Creates a LSTM network with a fully connected output layer.
    init_hidden() has to be called for every minibatch to reset the hidden state.

    Args:
        input_size (int): Length of input vector for each time step or the number of input features per time-step.
        hidden_size (int, optional): Size of hidden LSTM state
        num_layers (int, optional): Number of stacked LSTM modules
        dropout (float, optional): Dropout value to use inside LSTM and between
            LSTM layer and fully connected layer.
    """    
    def __init__(self, input_size, hidden_size = 128, num_layers = 1, bidirectional = True, dropout=0.2):
        super(LSTM_Model, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        if bidirectional is True:
            self.bi = 2
        else:
            self.bi = 1       
            
        # LSTM input dimension is: (batch_size, time_steps, num_features)
        # LSTM output dimension is: (batch_size, time_steps, hidden_size)        
        self.lstm = torch.nn.LSTM(input_size = input_size, 
                                  hidden_size= hidden_size,
                                  bidirectional = bidirectional,
                                  num_layers = num_layers, 
                                  batch_first= True, 
                                  dropout    = dropout) 
        
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_size*self.bi, 2)
                   
    def forward(self, x, hidden):
        self.lstm.flatten_parameters() # For deep copy   
        # output dimension is: (batch_size, time_steps, hidden_size)
        x = self.lstm(x, hidden)
#        pdb.set_trace()
        x = x[0][:, -1, :] # Take only last output of LSTM (many-to-one RNN)        
        # hidden_size contains all outputs [y] values (each LSTM cells produces one output)
        x = x.view(x.shape[0], -1) # Flatten to (batch_size, hidden_size)        
        x = self.dropout(x)
        x = self.fc(x)        
        return x

    def init_hidden(self, batch_size):
        '''
        Initializing the hidden layer.
        Call every mini-batch, since nn.LSTM does not reset it itself.
        '''
        h_0 = torch.zeros(self.num_layers*self.bi, batch_size, self.hidden_size)
        c_0 = torch.zeros(self.num_layers*self.bi, batch_size, self.hidden_size)
        
        return (h_0.to(dev), c_0.to(dev))

#%% CNN LSTM model
class CNN2DEncoder(torch.nn.Module):  
    """ Flexible 2D CNN 
    Example Usage:
        from nu_models import CNN_2DMod
        model = CNN_2DMod(kernel_size = [3, 3, 3, 3] , conv_channels = [1, 8, 16, 32])    
    """
    def __init__(self, 
                 kernel_size   = [3, 3, 3, 3], 
                 conv_channels = [1, 8, 16, 32],
                 dense_size    = 256,
                 dropout       = 0.1):            
        super(CNN2DEncoder, self).__init__()          
        self.cconv   = []  
        self.MaxPool = nn.MaxPool2d((1, 2), (1, 2))
        self.ReLU    = nn.ReLU()
        self.Dropout = nn.Dropout(dropout)        
        self.batchnorm = []       
        
        for jj in conv_channels:
            self.batchnorm.append(nn.BatchNorm2d(jj).cuda())               
        ii = 0        
        for in_channels, out_channels in zip(conv_channels, conv_channels[1:]):                           
            conv_i = torch.nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                                     kernel_size = kernel_size[ii], padding  = kernel_size[ii]//2)
            self.cconv.append(conv_i)                
            self.add_module('CNN_K{}_O{}'.format(kernel_size[ii], out_channels), conv_i)
            ii += 1  
            
    def forward(self, input):        
        for jj, conv_i in enumerate(self.cconv):
            input = conv_i(input)
            input = self.batchnorm[jj+1](input)
            input = self.ReLU(input)        
            input = self.MaxPool(input)              
        return input    
    
#%%
class CNNLSTM(torch.nn.Module):
    """
    Creates a LSTM network with a fully connected output layer.
    init_hidden() has to be called for every minibatch to reset the hidden state.

    Args:
        input_size (int): Length of input vector for each time step
        hidden_size (int, optional): Size of hidden LSTM state
        num_layers (int, optional): Number of stacked LSTM modules
        dropout (float, optional): Dropout value to use inside LSTM and between
            LSTM layer and fully connected layer.
    """    
    def __init__(self, input_size, cnn, hidden_size = 256, num_layers = 2, 
                 batch_size = 64, dropout = 0.1):
        super(CNNLSTM, self).__init__()
        
        self.cnn = cnn        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM input dimension is: (batch_size, time_steps, num_features)
        # LSTM output dimension is: (batch_size, time_steps, hidden_size)        
        
        self.lstm = torch.nn.LSTM(input_size  = input_size,
                                  hidden_size = hidden_size,
                                  num_layers  = num_layers,
                                  batch_first = True,
                                  dropout     = dropout)
        
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_size, 2)

    def forward(self, input, hidden):
        
        encoder = self.cnn(input)   
        # pdb.set_trace()
        self.lstm.flatten_parameters()

        batch_size, timesteps, H, W = encoder.size()
        r_in = encoder.view(batch_size, timesteps, -1)       
      
        # output dimension is: (batch_size, time_steps, hidden_size)
        x = self.lstm(r_in, hidden)[0][:, -1, :] # Take only last output of LSTM (many-to-one RNN)
        
        # hidden_size contains all outputs [y] values (each LSTM cells produces one output)
        x = x.view(x.shape[0], -1) # Flatten to (batch_size, hidden_size)        
        x = self.dropout(x)
        x = self.fc(x)        
        return x
        
    def init_hidden(self, batch_size):
        '''
        Initializing the hidden layer.
        Call every mini-batch, since nn.LSTM does not reset it itself.
        '''
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        if torch.cuda.is_available():
            return (h_0.to(dev), c_0.to(dev))
        else:
            return (h_0.to(dev), c_0.to(dev))

#%%            
class EEGNet(nn.Module):
    def __init__(self, 
                 time_samples, 
                 channels ):
      
        super(EEGNet, self).__init__()
        
        self.T       = time_samples      
        self.chans   = channels
        self.in_size = (1, time_samples, channels)
        
        self.layer1 = nn.Sequential(
                # Layer 1
                nn.Conv2d(1, 16, (1, self.chans), padding = 0),
                nn.BatchNorm2d(16, False),
                nn.ELU(),
                nn.Dropout(0.25) )        
        
        self.layer2and3 = nn.Sequential(
                # Layer 2
                nn.ZeroPad2d((16, 17, 0, 1)),
                nn.Conv2d(1, 4, (2, 32)),
                nn.BatchNorm2d(4, False),
                nn.ELU(),
                nn.Dropout(0.25),
                nn.MaxPool2d(2, 4),                
                
                # Layer 3
                nn.ZeroPad2d((2, 1, 4, 3)),
                nn.Conv2d(4, 4, (8, 4)),
                nn.BatchNorm2d(4, False),
                nn.Dropout(0.25),
                nn.MaxPool2d((2, 4)))
        
        self.flat_fts = self.get_out_dim(self.in_size)   
        self.fc1 = nn.Linear(self.flat_fts, 2)
        
    def get_out_dim(self, in_size):
        with torch.no_grad():
            # create a tensor 
            x = Variable(torch.ones(1, *self.in_size))
            x = self.layer1(x)
            x = x.permute(0, 3, 1, 2)
            x = self.layer2and3(x)
            x = int(np.prod(x.size()[1:])) 
            return x  
        
    def forward(self, x):        
        x = self.layer1(x)
        x = x.permute(0, 3, 1, 2)
        
        x = self.layer2and3(x)
        x = x.view(-1, self.flat_fts)        
        x = self.fc1(x)                
        return x
