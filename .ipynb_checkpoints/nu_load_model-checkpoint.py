# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:27:47 2019
@author: berdakh.abibullaev
"""
import pickle
import torch
from nu_models import CNN
model = CNN() 

#%% 
def loadfile(filename):    
    f = open(filename,'rb')
    data = pickle.load(f)
    f.close()
    return data     
#%%    
filename = 'Physionet__CNN_POOLEDC[1, 8, 16, 32, 64, 128]_K[5, 5, 5, 5, 5, 5]_0.75'
d = torch.load(filename)

#%% insert the weights
model.state_dict(d)    

#%%
for p in model.parameters():
    print(p.shape)
    
#%%
filename = 'PhysionetRR.pickle'

dd = loadfile(filename)