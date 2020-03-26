# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:03:25 2019
@author: Berdakh

This script can be used to train CNN model on pooled data.
"""
import torch 
import itertools
import pandas as pd 
import pickle 
import numpy as np

from nu_MIdata_loader import getTorch, EEGDataLoader
from nu_train_utils import train_model   
from nu_models import CNN2D

#%% to get a torch tensor 
get_data = getTorch.get_data 

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Your GPU device name :', torch.cuda.get_device_name())    
torch.manual_seed(0)

#%% get the data dimensionality 
dname = dict(
             BNCI2014004 = 'aBNCI2014004R.pickle',
             BNCI2014001 = 'aBNCI2014001R.pickle',
             Weibo2014   = 'aWeibo2014R.pickle',
             Physionet   = 'aPhysionetRR.pickle'             
             )
      
#%% Hyperparameter settings
num_epochs = 1 
learning_rate = 1e-3
weight_decay = 1e-4  

batch_size = 64
verbose = 1

augmentdata = dict(std_dev = 0.01,  multiple = 1)

#%% one should run this script twice with ConvDown = True or False to have different convolutional layer patterns
# as defined below by params dictio2. 
ConvDOWN = False   # change this option 

fs = 80

crop_length = 1.5 #seconds 
crop = dict(fs = fs, crop_length = crop_length)

class1, class2 = 'left_hand', 'right_hand'
s = list(range(108))
 
#%% The main loop starts here 
# for each dataset in dname train CNN on pooled data 
for itemname, filename in dname.items():
    print('working with', filename)
    iname = itemname + '__'    
        
    d = EEGDataLoader(filename, class_name = [class1, class2])    
    d1 = d.load_pooled(s, normalize = True, crop = crop, test_size = 0.01, augmentdata = augmentdata )
        
    #% used to save the results table 
    results = {}        
    table = pd.DataFrame(columns = ['Train_Acc', 'Val_Acc', 'Test_Acc', 'Epoch'])       
    
    # get data 
    dat = get_data(d1, batch_size, image = True, lstm = False, raw = False)   
    
    dset_loaders = dat['dset_loaders']
    dset_sizes   = dat['dset_sizes']   
    
    #% identify input size (channel x timepoints)
    timelength = d1['xtest'].shape[2]
    chans      = d1['xtest'].shape[1]
    input_size = (1, chans, timelength)    
    
    # define kernel size in terms of ms length 
    timE = 100 #ms
    ker = timE*fs//1000    
    
    # ker = 8 #timelength//chans         
    # convolution parameters 
    a, a1 = 3, 1
    b, b1 = 3, 3
    c, c1 = 3, 5       
    
    if ConvDOWN:            
        params = {'conv_channels': [
                                    [1, 16, 8],                                               
                                    [1, 32, 16, 8],
                                    [1, 64, 32, 16, 8],
                                    [1, 128, 64, 32, 16, 8],
                                    [1, 256, 128, 64, 32, 16, 8]                                     
                                    ],                                    
    					
                  'kernel_size':    [[(a, a1*ker), (a, a1*ker), (a, a1*ker),(a, a1*ker),(a, a1*ker),(a, a1*ker)],
                                     [(b, b1*ker), (b, b1*ker), (b, b1*ker),(b, b1*ker),(b, b1*ker),(b, b1*ker)],
                                     [(c, c1*ker), (c, c1*ker), (c, c1*ker),(c, c1*ker),(c, c1*ker),(c, c1*ker)]]                                                                      
                  }                      
    else:                      
        params = {'conv_channels': [
                                    [1, 8, 16],                                                  
                                    [1, 8, 16, 32],
                                    [1, 8, 16, 32, 64],
                                    [1, 8, 16, 32, 64, 128],
                                    [1, 8, 16, 32, 64, 128, 256]
                                    ],      		
        					
                  'kernel_size':    [[(a, a1*ker), (a, a1*ker), (a, a1*ker),(a, a1*ker),(a, a1*ker),(a, a1*ker)], 
                                     [(b, b1*ker), (b, b1*ker), (b, b1*ker),(b, b1*ker),(b, b1*ker),(b, b1*ker)],
                                     [(c, c1*ker), (c, c1*ker), (c, c1*ker),(c, c1*ker),(c, c1*ker),(c, c1*ker)]]                     
                  }     
                  
    keys = list(params)
 
    for values in itertools.product(*map(params.get, keys)):             
        d = dict(zip(keys, values))
        description = 'C{}_K{}'.format(d['conv_channels'], d['kernel_size'][0])    
        print('\n\n##### ' + description + ' #####')
        
        # Define the model
        model = CNN2D(input_size    = input_size,
                      kernel_size   = d['kernel_size'], 
                      conv_channels = d['conv_channels'],
                      dense_size    = 256,
                      dropout       = 0.5)        
        
        print("Model architecture >>>", model)
        
        # optimizer and the loss function definition 
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        
        # move the model to GPU/CPU
        model.to(dev)  
        criterion.to(dev)       
            
        #******** Training loop *********    
        best_model, train_losses, val_losses, train_accs, val_accs , info = train_model(model, dset_loaders, 
                                                                                        dset_sizes, criterion,
                                                                                        optimizer, dev, 
                                                                                        lr_scheduler=None, 
                                                                                        num_epochs=num_epochs,                                                                                     
                                                                                        verbose = verbose)    
        
        
        #1) save your model in your hard drive 
        #2) load it and test it on test examples 
        
        # evaluate the best model
        test_samples = 150
        x_test = dat['test_data']['x_test'][:test_samples,:,:,:] 
        y_test = dat['test_data']['y_test'][:test_samples] 
        
        preds = best_model(x_test.to(dev)) 
        preds_class = preds.data.max(1)[1]
        
        # Test Accuracy 
        corrects = torch.sum(preds_class == y_test.data.to(dev))     
        test_acc = corrects.cpu().numpy()/x_test.shape[0]
        
        print("Test Accuracy :", test_acc) 
        
        # Save results       
        tab = dict(Train_Acc  = train_accs[info['best_epoch']],
                   Val_Acc    = val_accs[info['best_epoch']],   
                   Test_Acc   = test_acc, 
                   Epoch      = info['best_epoch'] + 1)         
        
        table.loc[description] = tab  
        val_acc = np.max(val_accs)
        
        print(table)
        results[description] = dict(train_accs = train_accs, 
                                    val_accs =  val_accs,                                
                                    ytrain = info['ytrain'], 
                                    yval= info['yval'])      
        
        fname = iname + 'CNN_POOLED' + description + '_' + str(val_acc)[:4]
        torch.save(best_model.state_dict(), fname) 
        
    # save all the results in one file 
    result_cnn = dict(table = table)
    fname2 = iname + "__CNN_POOLED_RESULTS_ALL_" + description + str(np.random.randint(77)) + 'ConvDown' + str(ConvDOWN)         
    
    with open(fname2, 'wb') as fp:
        pickle.dump(result_cnn, fp)   
    