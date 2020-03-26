#!/usr/bin/env python
# coding: utf-8
"""
This script can be used to train EEGNet model. Note that we used the EEGNet implementation 
provided via the brain decode toolbox available at https://robintibor.github.io/braindecode/.

You can install the library via >>> pip install braindecode
"""

import logging
import importlib
importlib.reload(logging)  

log = logging.getLogger()
log.setLevel('INFO')
import sys

logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)
import torch 
import pandas as pd 
import pickle 

# import helper functions
from nu_MIdata_loader import getTorch, EEGDataLoader
from nu_train_utils import train_model   
get_data = getTorch.get_dataEEGnet 

# set device type as CPU or GPU 
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Your GPU device name :', torch.cuda.get_device_name())    

from braindecode.models.eegnet import EEGNet 
Model = EEGNet

#%%  
dname = dict(
             BNCI2014004 = 'aBNCI2014004R.pickle',
             BNCI2014001 = 'aBNCI2014001R.pickle',
             Weibo2014   = 'aWeibo2014R.pickle',
             Physionet   = 'aPhysionetRR.pickle'             
             )

#%% EEGNet hyperparameter settings
num_epochs = 150 
learning_rate = 1e-3
weight_decay = 1e-4  

batch_size = 64 
verbose = 2
n_classes = 2      
saveResults = True

#%% The main loop starts here 
# for each dataset in dname EEGNet on pooled data 
for itemname, filename in dname.items():
    print('working with', filename)
    iname = itemname + '__'    
       
    d = EEGDataLoader(filename)
    
    # subject data indicies 
    s = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]         
    d1 = d.load_pooled(s)
        
    #% identify input size (channel x timepoints)
    timelength = d1['xtest'].shape[2]
    chans = d1['xtest'].shape[1]        
        
    dat = get_data(d1, batch_size, lstm = False, image = True)        
    dset_loaders = dat['dset_loaders']
    dset_sizes   = dat['dset_sizes']    
    
    #% used to save the results table 
    table = pd.DataFrame(columns = ['Train_Acc', 'Val_Acc', 'Test_Acc', 'Epoch'])
    results = {}     

    model = Model(in_chans=chans, n_classes=n_classes,
              input_time_length=timelength,
              final_conv_length='auto').create_network()  
    
    print("Model architecture >>>", model)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.to(dev)  
    criterion.to(dev)       
            
    #******** Training loop *********    
    best_model, train_losses, val_losses, train_accs, val_accs , info = train_model(model, dset_loaders, 
                                                                                    dset_sizes, 
                                                                                    criterion, optimizer,
                                                                                    dev, lr_scheduler=None,
                                                                                    num_epochs=num_epochs,                                                                                     
                                                                                    verbose = verbose)    
    
    #%******* TEST THE BEST MODEL ***********
    # One could immediately test the performance of the best model on test set as shown below:
    x_test = dat['test_data']['x_test'] 
    y_test = dat['test_data']['y_test'] 
    #************************
    
    preds = best_model(x_test.to(dev)) 
    preds_class = preds.data.max(1)[1]
    
    # calculate the accuracy
    corrects = torch.sum(preds_class == y_test.data.to(dev))     
    test_acc = corrects.cpu().numpy()/x_test.shape[0]
    print("Test Accuracy :", test_acc)    
    
    if saveResults: # save the model and the performance metrics        
        tab = dict(Train_Acc  = train_accs[info['best_epoch']],
                   Val_Acc    = val_accs[info['best_epoch']],   
                   Test_Acc   = test_acc, 
                   Epoch      = info['best_epoch'] + 1)  
        
        description = 'EEGnet_pooled'
        table.loc[description] = tab  
        
        results[description] = dict(train_accs = train_accs, val_accs =  val_accs,                                
                                    ytrain = info['ytrain'], yval= info['yval'])      
        
        fname = iname + 'EEGnet_POOLED' + description + '_' + str(info['best_acc'])[:4]+ "__" + str(test_acc)
        torch.save(best_model.state_dict(), fname)         
        
        result_cnn = dict(table = table, results = results)
        fname2 = iname + "__EEGnet_POOLED_RESULTS_ALL"         
        
        with open(fname2, 'wb') as fp:
            pickle.dump(result_cnn, fp, protocol=pickle.HIGHEST_PROTOCOL)        
        