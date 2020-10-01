"""
Created on Wed Feb 27 11:52:10 2019
@author: berdakh.abibullaev
"""
import pickle 
import numpy as np
import torch  
 
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader  
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin 
 
#%% use sklearn standard scaler 
class SKStandardScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)    
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self
    
    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X
    
    def _flatten(self, X):        
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):        
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X
    
#%% 
def augment_dataset(X, Y, std_dev, multiple):
    """
    Augments the size of the dataset by introducing unbiased gaussian noise.
    Resulting dataset is 'multiple' times bigger than original.
    Args:
        X (torch.FloatTensor): Input training data
        Y (torch.FloatTensor): Target training data

        std_dev (float): Standard deviation of gaussian noise to apply
        multiple (int): Factor by how much the dataset should be bigger
    """
#    pdb.set_trace()
    new_X, new_Y = X.clone(), Y.clone()
    
    for i in range(multiple-1):        
        augmented_input  = X + torch.zeros_like(X).normal_(0, std_dev)
        new_X  = torch.cat((new_X, augmented_input))
        new_Y = torch.cat((new_Y, Y))
        
    return new_X, new_Y

#%%
def crop_data1(fs, crop_length, xdata, ylabel):       

    xpercent = 50 
    xoverlap = crop_length*xpercent/100    
    desired_length = np.int(fs*crop_length)
    overlap = np.int(fs*xoverlap) 
     
    number_splits  = xdata.shape[-1]//desired_length
    
    tstart = 0
    tstop = desired_length     
    #% needed to copy multiple times
    t = 3 - crop_length    
    for ii in np.arange(number_splits + t):       
        if ii == 0:
            tstart = tstart    
            tstop  = tstart + desired_length + overlap
            
            Xi = xdata[:,:,tstart:tstop]
            Yi = ylabel    
#            print(tstart/fs, tstop/fs)    
#            print('X::', Xi.shape, '-- Y::',  Yi.shape)    
        else:
            try:                
                tstart = tstart + desired_length  
                tstop  = tstart + desired_length + overlap
                # concatenate 
                Xi = torch.cat([Xi, xdata[:,:,tstart:tstop]])
                Yi = torch.cat([Yi, ylabel])                      
            except:
                pass             
    return Xi, Yi  

#%%    
def crop_data(fs, crop_length, xdata, ylabel):       
    # this function takes torch tensors and performs cropping along the time points
    # and generates mulitple cropped examples 
    # fs = 128
    # crop_length = 1 # seconds     
    # xdata = tensor dataset with [samples x channels x time pints]
    print('::: Cropping :::')
    
    desired_length = np.int(fs*crop_length)
    number_splits  = xdata.shape[-1]//desired_length
    
    tstart = 0
    tstop = desired_length    
    
    for ii in np.arange(number_splits):       
        if ii == 0:
            Xi = xdata[:,:,tstart:tstop]
            Yi = ylabel        
        else:        
            tstart = tstart + desired_length    
            tstop  = tstart + desired_length

            Xi = torch.cat([Xi, xdata[:,:,tstart:tstop]])
            Yi = torch.cat([Yi, ylabel])        
#        print(tstart, tstop)    
#        print('X::', Xi.shape, '-- Y::',  Yi.shape)         
    return Xi, Yi

#%% ####################################
class EEGDataLoader(object):         
    def __init__(self, filename, datapath = "", class_name = ['left_hand', 'right_hand'] ):             
        self.filename = filename
        self.datapath = datapath
        self.class_name = class_name        
        
    def load_pooled(self, subjectIndex, normalize = False, crop = None, 
                    test_size = None, augmentdata = None):         
        """Loads all the data from the EEG dataset.              
        returns dictionary of:
            X_train, X_valid, X_test: np.array of shape (samples, channel, times), data features
            y_train: np.array of shape (samples), data labels
               """    
        with open(self.filename, 'rb') as handle:
            b = pickle.load(handle)       
        #% extract positive and negative classes  
        pos, neg = [], []           
        # filetype is used for correct indexing in NU and MOABB datasets 
        # default MNE keys are set to:        
        target, nontarget = self.class_name[0], self.class_name[1]
             
        for ii in subjectIndex:
            try:                    
                pos.append(b[ii][target].get_data())
                neg.append(b[ii][nontarget].get_data())            
            except Exception:
                pass                       
        # prepare the pooled data AND concatenate data from all subjects 
        s1pos, s1neg = pos[-1], neg[-1] # get the data from the last subject in the list 
        for jj in range(len(pos)-1): # all subject but the last one 
            s1pos = np.concatenate([s1pos, pos[jj]])
            s1neg = np.concatenate([s1neg, neg[jj]])     
            
        # get the labels and construct data array from all subjects 
        X = np.concatenate([s1pos.astype('float32'), s1neg.astype('float32')])        
        Y = np.concatenate([np.ones(s1pos.shape[0]).astype('float32'), 
                            np.zeros(s1neg.shape[0]).astype('float32')])        
        # normalization 
        if normalize:                                    
            scaler = SKStandardScaler()
            X = scaler.fit_transform(X)         
     
        x_rest, x_test, y_rest, y_test =\
            train_test_split(X, Y, test_size = test_size, random_state=42, stratify = Y)
        
        x_train, x_valid, y_train, y_valid =\
            train_test_split(x_rest, y_rest, test_size = 0.20, random_state = 42, stratify = y_rest)                   
    
        # Convert to Pytorch tensors
        X_train, X_valid, X_test = map(torch.FloatTensor, (x_train, x_valid, x_test))
        y_train, y_valid, y_test = map(torch.FloatTensor, (y_train, y_valid, y_test))        

        if augmentdata['multiple']:
           X_train, y_train = augment_dataset(X_train, y_train, augmentdata['std_dev'], augmentdata['multiple'])
     
        if crop['crop_length']:
           fs = crop['fs']
           crop_length = crop['crop_length']
           
           X_train, y_train =  crop_data1(fs, crop_length, X_train, y_train)
           X_valid, y_valid =  crop_data1(fs, crop_length, X_valid, y_valid)
           X_test,  y_test  =  crop_data1(fs, crop_length, X_test, y_test)

        return dict(xtrain = X_train, xvalid = X_valid, xtest = X_test,
                    ytrain = y_train, yvalid = y_valid, ytest = y_test)
        
    
    #%% returns subject specific data dictionary with xtrain, xvalid, xtest     
    def subject_specific(self, subjectIndex, normalize = False, crop = None, 
                         test_size = None, augmentdata = None):   

        with open(self.filename, 'rb') as handle:
            b = pickle.load(handle)       
        
        # extract positive and negative classes  
        pos, neg = [], []      
        datt = []         
       # filetype is used for correct indexing in NU and MOABB datasets 
        # default MNE keys are set to: 
        target, nontarget = self.class_name[0], self.class_name[1]
                     
        if len(subjectIndex) > 1:     
            try:  # so that subject index could exceed the number of available datasets                   
                for jj in subjectIndex:                
                    print('Loading subjects:', jj)   
                    dat = b[jj]                                     
                    pos.append(dat[target].get_data())
                    neg.append(dat[nontarget].get_data())  
            except Exception as err:
                print(err)
        else:
            print('Loading subject:', subjectIndex[0]+1)  
            dat = b[subjectIndex[0]]
            pos.append(dat[target].get_data())
            neg.append(dat[nontarget].get_data())  
               
        # subject specific upsampling 
        for ii in range(len(pos)):              
            X, Y = [], []                  
            X = np.concatenate([pos[ii].astype('float32'), neg[ii].astype('float32')])            
            Y = np.concatenate([np.ones(pos[ii].shape[0]).astype('float32'), np.zeros(neg[ii].shape[0]).astype('float32')])            
            
            #% normalization 
            if normalize:
                scaler = SKStandardScaler()
                X = scaler.fit_transform(X)            
               
            x_rest, x_test, y_rest, y_test =\
                train_test_split(X, Y, test_size = test_size, random_state=42, stratify = Y)
            
            x_train, x_valid, y_train, y_valid =\
                train_test_split(x_rest, y_rest, test_size = 0.20, random_state = 42, stratify = y_rest)                   
                
           # Convert to Pytorch tensors
            X_train, X_valid, X_test = map(torch.FloatTensor, (x_train, x_valid, x_test))
            y_train, y_valid, y_test = map(torch.FloatTensor,  (y_train, y_valid, y_test)) 

            if augmentdata['multiple']:
                print('Augmenting the data')                               
                X_train, y_train =  augment_dataset(X_train, y_train, augmentdata['std_dev'], augmentdata['multiple'])      
              
            if crop['crop_length']: # this takes the data and performs cropping along the time series                
               fs = crop['fs']
               crop_length = crop['crop_length']               
               X_train, y_train =  crop_data1(fs, crop_length, X_train, y_train)
               X_valid, y_valid =  crop_data1(fs, crop_length, X_valid, y_valid)
               X_test,  y_test  =  crop_data1(fs, crop_length, X_test, y_test)
                         
            datt.append(dict(xtrain = X_train, xvalid = X_valid, xtest = X_test,
                             ytrain = y_train, yvalid = y_valid, ytest = y_test))                   
        return datt   

#%% Get data loader  
class getTorch(object):    
    def __init__(self):
        pass     
    @staticmethod      
    def get_data(data, batch_size, lstm, image, raw):         
        """ This function takes data that is obtained from sklearn 
        train_test_split and wraps with pytorch dataloaders            
        
        Input: 
            data : is dictionary with the following structure        
                  dict_keys(['xtrain', 'xvalid', 'xtest', 'ytrain', 'yvalid', 'ytest'])
            where:                 
            *xtrain, xvalid, xtest : [trials x channels x time_samples] is ndarray         
            *labels: 'ytrain', 'yvalid', 'ytest'
        
        Output: 
            pytorch dataloader dictionary object with  [xtrain, xvalid, xtest]
            
        Options:
            if LSTM = TRUE,  data will be reshaped to be used with LSTM             
            if IMAGE = TRUE then data is reshaped as an gray scale image 
            if RAW = TRUE,  the original data is returned without reshaping             
        """       
        
        # Input data is a dictionary 
        x_train, y_train = data['xtrain'], data['ytrain']
        x_valid, y_valid = data['xvalid'], data['yvalid'] 
        x_test,  y_test  = data['xtest'],  data['ytest']         
        
        if lstm: # re-arranges the data to use with LSTM 
          x_train = x_train.permute(0, 2, 1) 
          x_valid = x_valid.permute(0, 2, 1)    
          x_test  = x_test.permute(0, 2, 1)      
          
        if image: # this option will reshape the input as a gray scale image 
          x_train = torch.unsqueeze(x_train, dim = 1) 
          x_valid = torch.unsqueeze(x_valid, dim = 1)
          x_test  = torch.unsqueeze(x_test, dim = 1)
        
        print('Input data shape', x_train.shape)   
        ##############################################                   
        # TensorDataset      
        train_dat    = TensorDataset(x_train, y_train) 
        val_dat      = TensorDataset(x_valid, y_valid)    
        
        ##############################################
        train_loader = DataLoader(train_dat, batch_size = batch_size, shuffle = True, drop_last=False)
        val_loader   = DataLoader(val_dat,   batch_size = batch_size, shuffle = True, drop_last=False)
            
        if raw: # get the raw inputs (no TensorDataset nor DataLoader used)
          out = dict(train_input  = x_train, 
                     x_valid      = x_valid, 
                     train_target = y_train, 
                     y_valid      = y_valid,
                     test_data    = {'x_test' : x_test, 'y_test' : y_test})   
        
        else: # return data loaders 
          out = dict(dset_loaders = {'train': train_loader, 'val': val_loader}, 
                     dset_sizes  =  {'train': len(x_train), 'val': len(x_valid)},
                     test_data   =  {'x_test' : x_test, 'y_test' : y_test})          
        return out 
    
    
    @staticmethod      
    def get_dataEEGnet(data, batch_size, lstm, image):         
        """ This function takes data that is obtained from sklearn 
        train_test_split and wraps with pytorch dataloaders            
        
        Input: 
            data : is dictionary with the following structure        
                  dict_keys(['xtrain', 'xvalid', 'xtest', 'ytrain', 'yvalid', 'ytest'])
            where:                 
            *xtrain, xvalid, xtest : [trials x channels x time_samples] is ndarray         
            *labels: 'ytrain', 'yvalid', 'ytest'
        
        Output: 
            pytorch dataloader dictionary object with  [xtrain, xvalid, xtest]
            
        Options:
            if IMAGE = TRUE then data is reshaped as an gray scale image 
        """   
        # Input data is a dictionary 
        x_train, y_train = data['xtrain'], data['ytrain']
        x_valid, y_valid = data['xvalid'], data['yvalid'] 
        x_test,  y_test  = data['xtest'],  data['ytest']         
                 
        if lstm: # re-arranges the data to use with LSTM 
          x_train = x_train.permute(0, 2, 1) 
          x_valid = x_valid.permute(0, 2, 1)    
          x_test  = x_test.permute(0, 2, 1)      
                    
        if image: # this option will reshape the input as a gray scale image 
          x_train = torch.unsqueeze(x_train, dim = -1) 
          x_valid = torch.unsqueeze(x_valid, dim = -1)
          x_test  = torch.unsqueeze(x_test, dim =  -1)
                                     
        print('Input data shape', x_train.shape)   
        # TensorDataset      
        train_dat    = TensorDataset(x_train, y_train) 
        val_dat      = TensorDataset(x_valid, y_valid)    
        train_loader = DataLoader(train_dat, batch_size = batch_size, shuffle = True)
        val_loader   = DataLoader(val_dat,   batch_size = batch_size, shuffle = False)
                     
        return dict(dset_loaders = {'train': train_loader, 'val': val_loader}, 
                    dset_sizes   = {'train': len(x_train), 'val': len(x_valid)},
                    test_data    = {'x_test' : x_test, 'y_test' : y_test})         