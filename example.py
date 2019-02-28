#!/usr/bin/env python
# coding: utf-8

# Code to run SOCNN model on sample toy dataset.

# In[2]:


import nnts
import pandas as pd
import numpy as np
import os
#from nnts.utils import *


# Defining sample data frame:
# - column A enumerates entries, B contains random binomial variables, 
# - columns C and D contain random noise
# - column E is a sum of last 10 values of B multiplied by D

# In[2]:


df = pd.DataFrame({'A': np.arange(1000),
                   'B': (np.random.rand(1000)> .5) * 1.0, 
                   'C': np.random.rand(1000), 
                   'D': np.random.rand(1000)})
df['E'] = df['B'] * df['D'] 
df['E'] = np.cumsum(df['E'])
df.loc[10:, 'E'] -= np.array(df['E'][:-10])
print(df.head(20))


# Saving data frame to csv. Note that csv format is required by the model to read from.

# In[3]:


dataset_file = os.path.join(os.path.dirname(__file__), 'data', 'example1.csv')
df.to_csv(dataset_file)


# Speifying parameters for training. For each of the keyword parameters we define a list of values for grid search.
# 
# We want to train models that predict column A given B, C and D, and A and E, given B, C and D.  

# In[4]:


param_dict = dict(
    # input parameters
    input_column_names = [['B', 'C', 'D']],            # input columns 
    target_column_names = [['E'], ['A', 'E']],         # target columns
    diff_column_names = [[]],                          # columns to take first difference of   
    verbose = [2],                  # verbosity
    train_share = [(.8, .9, 1.)],   # delimeters of the training and validation shares
    input_length = [20],            # input length (1 - stateful lstm)
    output_length = [1],            # no. of timesteps to predict (only 1 impelemented)
    batch_size = [16],              # batch size
    objective=['regr'],             # only 'regr' (regression) implemented
    #training_parameters
    patience = [5],                 # no. of epoch after which learning rate will decrease if no improvement
    reduce_nb = [2],                # no. of learning rate reductions
    lr = [.01],                     # initial learning rate
    clipnorm = [1.0],               # max gradient norm
    #model parameters
    norm = [10],                    # max norm for fully connected top layer's weights
    filters = [8],                  # no. of convolutional filters per layer
    act = ['leakyrelu'],            # activation ('linear', 'relu', 'sigmoid', 'tanh', 'leakyrelu')
    kernelsize = [[1, 3], 1, 3],    # kernel size (if list of ints passed kernel size changes successively in consecutive layers)
    layers_no = [{'sigs': 5, 'offs': 1}],                # no. of layers for significance and offset sub-networks             
    architecture = [{'softmax': True, 'lambda': False}], # final activation: lambda=True indicates softplus   
    nonnegative = [False],          # if True, apply only nonnegative weights at the top layer
    connection_freq = [2],          # vertical connection frequency for ResNet
    aux_weight = [0.1],             # auxilllary loss weight
    shared_final_weights = [False], # if True, same weights of timesteps for all dimentions are trained
    resnet = [False],               # if True, adds vertical connections
)


# Model class import. Other benchmark models such as LSTM, CNN or ResNet can be found in nnts.models module.

# In[5]:


from nnts.models import SOCNN


# Running the grid seach. Results will be pickled in 'nnts/results' directory.

# In[6]:


save_file = os.path.join(os.path.dirname(__file__), 'results', 'example_model.pkl')
runner = nnts.utils.ModelRunner(param_dict, [dataset_file], save_file)

results = runner.run(SOCNN.SOCNNmodel, log=False, limit=1)


# Results are stored in a list of dictionaries. Each dictionary stores parameters and results (loss function values troughout consecutive epochs) that correspond to each single model fitted during training.

# In[7]:


for i, r in enumerate(results[:3]):
    print('########## results[%d] ##########' % i)
    for k, v in r.items():
        print(str(k).rjust(25) + ': ' + str(v)[:80] + ' ...' * (len(str(v)) > 80 ))
    print('\n\n')

