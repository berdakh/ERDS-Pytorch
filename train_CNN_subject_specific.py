#%% Subject specific CNN %%         
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:03:25 2019
@author: Berdakh

This script can be used to train CNN model on subject specific data.
"""
import numpy as np
import torch 
import pandas as pd 
import pickle 

from nu_MIdata_loader import getTorch, EEGDataLoader
from nu_train_utils import train_model   

# to get a torch tensor 
get_data = getTorch.get_data 
 

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Your GPU device name :', torch.cuda.get_device_name())  
# C[32,16,8]_K[7,7,5]

#%%
dname = dict(BNCI2014004 = 'BNCI2014004R.pickle',
             BNCI2014001 = 'BNCI2014001R.pickle',
             Weibo2014   = 'Weibo2014R.pickle',
             Physionet   = 'PhysionetRR.pickle')


#%% Hyperparameter settings 
from nu_models import CNN2D

fs = 80
num_epochs = 150
batch_size = 64
verbose = 1
learning_rate = 1e-3
weight_decay = 1e-4  

savemodel  = False
freezeCNN  = False 
initialize = False

#%%
# sampling frequency  
# subject list 
s = list(range(110))

#s = [96, 92, 35, 99, 80, 10, 8, 0, 108 ]
for itemname, filename in dname.items():
    print('working with', filename)
    iname = itemname + '__'

    if itemname == 'BNCI2014004':
            kernel_size = [7, 7, 5, 5, 3, 3, 3]	
            conv_chan   = [1, 64, 32, 16, 8]            
            crop_length = None          
            augmentdata = dict(std_dev = 0.01,  multiple = 2)
            
    elif itemname == 'Weibo2014':
            kernel_size = [3, 3, 3, 3, 3, 3, 3]	
            conv_chan   = [1, 64, 32, 16, 8]            
            crop_length = 2
            augmentdata = dict(std_dev = 0.01,  multiple = 2)
 
    elif itemname == 'Physionet':        
            kernel_size = [5, 5, 5, 5, 5, 5, 5]	
            conv_chan   = [1, 8, 16, 32, 64]            
            crop_length = 2             
            augmentdata = dict(std_dev = 0.01,  multiple = 2)
            
    print('>>>>>>>> default model <<<<<<<<<<') 
    #kernel_size = [(3, 8), (3, 8), (3, 8), (3, 8), (3, 8), (3, 8)]	
    #conv_chan   = [1, 128, 64, 32, 16, 8]   
    
    crop = dict(fs = fs, crop_length = crop_length) 
    
    d = EEGDataLoader(filename, class_name = ['left_hand', 'right_hand'] )  
    
    data = d.subject_specific(s, normalize = True, crop = crop, 
                              test_size    = 0.1, augmentdata  = augmentdata)    
    
    #% input size (channel x timepoints)
    timelength = data[0]['xtrain'].shape[2]
    chans      = data[0]['xtrain'].shape[1]
    input_size = (1, chans, timelength)    
    
    datum = {}
    for ii in range(len(data)):
      datum[ii] = get_data(data[ii], batch_size, image = True, lstm = False, raw = False)
    
    results = {}
    table = pd.DataFrame(columns = ['Train_Acc', 'Val_Acc', 'Test_Acc', 'Epoch'])
     
    model = CNN2D(input_size    = input_size,
                  kernel_size   = kernel_size,  
                  conv_channels = conv_chan,
                  dense_size    = 256,
                  dropout       = 0.25)  
            
    model.to(dev)
    # if initialize: model.state_dict(model_weights)        
    print(model)
    
    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()        
    criterion.to(dev) 
                
    if freezeCNN:            
        for name, param in model.named_parameters():            
            if name != 'fc2.weight' or 'fc2.bias':# or 'CNN_K5_O128' or 'CNN_K5_O64' or 'CNN_K5_O32' or 'CNN_K5_O16':                     
                param.requires_grad = False
        
    #% for each subject in a given data type perform model selection
    for subjectIndex in datum:   

        dset_loaders = datum[subjectIndex]['dset_loaders']
        dset_sizes   = datum[subjectIndex]['dset_sizes']      
          
        print('::: processing subject :::', subjectIndex)      
        #***************** Training loop ********************
        best_model, train_losses, val_losses, train_accs, val_accs , info = train_model(model, dset_loaders, 
                                                                                        dset_sizes,  
                                                                                        criterion,optimizer,
                                                                                        dev, lr_scheduler=None, 
                                                                                        num_epochs=num_epochs, 
                                                                                        verbose = verbose)      
        test_samples = 500
        x_test = datum[subjectIndex]['test_data']['x_test']#[:test_samples,:,:,:] 
        y_test = datum[subjectIndex]['test_data']['y_test']#[:test_samples]
        print('test_size -> ', len(y_test))
          
        preds = best_model(x_test.to(dev))    
        preds_class = preds.data.max(1)[1]
          
        corrects = torch.sum(preds_class == y_test.data.to(dev))     
        test_acc = corrects.cpu().numpy()/x_test.shape[0]  
        print("Test Accuracy :", test_acc)    
        
        # save results       
        tab = dict(Train_Acc  = train_accs[info['best_epoch']],
                   Val_Acc    = val_accs[info['best_epoch']],   
                   Test_Acc   = test_acc, Epoch = info['best_epoch'] + 1)      
        
        table.loc[subjectIndex+1] = tab 
        
        if savemodel:                
            modelname = iname +'S'+ str(subjectIndex)+ 'TT_CNN'+  '_' + str(test_acc)[:6]               
            torch.save(best_model.state_dict(), modelname) 
        
        results[subjectIndex] = dict(train_losses = train_losses, val_losses = val_losses,
                                     train_accs = train_accs,     val_accs =  val_accs,                                
                                     ytrain = info['ytrain'],     yval= info['yval'])  
        resultat = dict(table = table, results = results)  
        
        fname2 = iname + "TT__CNN_subspe_results" +'CROP'+ str(crop['crop_length'])+ str(np.random.randint(111))       

    with open(fname2, 'wb') as fp:
        pickle.dump(resultat, fp)
        
    print(table) 
    print(table.mean())