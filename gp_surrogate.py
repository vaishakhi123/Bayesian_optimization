import numpy as np
import torch
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import GPyOpt
import GPy
import os
import matplotlib as mpl
import matplotlib.tri as tri
from collections import Counter
import matplotlib.ticker as ticker
from sklearn import preprocessing
import random
from scipy.stats import norm
import time
from sklearn.ensemble import RandomForestRegressor
import copy


# added libraries
from tqdm import tqdm

# go to directory where datasets reside
# load a dataset
# dataset names = ['Crossed barrel', 'Perovskite', 'AgNP', 'P3HT', 'AutoAM']
dataset_name = 'P3HT'
raw_dataset = pd.read_csv('datasets/' + dataset_name + '_dataset.csv')
print(f'Dataset selected: {dataset_name}')
raw_dataset

feature_name = list(raw_dataset.columns)[:-1]
feature_name

objective_name = list(raw_dataset.columns)[-1]
objective_name

# for P3HT/CNT, Crossed barrel, AutoAM, their original goals were to maximize objective value.
# here, we add negative sign to all of its objective values here 
# because default BO in the framework below aims for global minimization
# only P3HT/CNT, Crossed barrel, AutoAM need this line; Perovskite and AgNP do not need this line.
ds = copy.deepcopy(raw_dataset) 
ds[objective_name] = -raw_dataset[objective_name].values
ds

# for some datasets, each input feature x could have been evaluated more than once.
# to perform pool-based active learning, we need to group the data by unique input feature x value. 
# for each unique x in design space, we only keep the average of all evaluations there as its objective value
ds_grouped = ds.groupby(feature_name)[objective_name].agg(lambda x: x.unique().mean())
ds_grouped = (ds_grouped.to_frame()).reset_index()
ds_grouped

# these are the input feature x and objective value y used in framework
X_feature = ds_grouped[feature_name].values

y = np.array(ds_grouped[objective_name].values)

assert len(ds_grouped) == len(X_feature) == len(y)

# total number of data in set
N = len(ds_grouped)
print(N)

# here are some parameters of the framework, feel free to modify for your own purposes

# number of ensembles. in the paper n_ensemble = 50.
n_ensemble = 50
# number of initial experiments
n_initial = 2
# number of top candidates, currently using top 5% of total dataset size
n_top = int(math.ceil(len(y) * 0.05))
# the top candidates and their indicies
top_indices = list(ds_grouped.sort_values(objective_name).head(n_top).index)

# random seeds used to distinguish between different ensembles
# there are 300 of them, but only first n_ensemble are used
seed_list = [4295, 8508, 326, 3135, 1549, 2528, 1274, 6545, 5971, 6269, 2422, 4287, 9320, 4932, 951, 4304, 1745, 5956, 7620, 4545, 6003, 9885, 5548, 9477, 30, 8992, 7559, 5034, 9071, 6437, 3389, 9816, 8617, 3712, 3626, 1660, 3309, 2427, 9872, 938, 5156, 7409, 7672, 3411, 3559, 9966, 7331, 8273, 8484, 5127, 2260, 6054, 5205, 311, 6056, 9456, 928, 6424, 7438, 8701, 8634, 4002, 6634, 8102, 8503, 1540, 9254, 7972, 7737, 3410, 4052, 8640, 9659, 8093, 7076, 7268, 2046, 7492, 3103, 3034, 7874, 5438, 4297, 291, 5436, 9021, 3711, 7837, 9188, 2036, 8013, 6188, 3734, 187, 1438, 1061, 674, 777, 7231, 7096, 3360, 4278, 5817, 5514, 3442, 6805, 6750, 8548, 9751, 3526, 9969, 8979, 1526, 1551, 2058, 6325, 1237, 5917, 5821, 9946, 5049, 654, 7750, 5149, 3545, 9165, 2837, 5621, 6501, 595, 3181, 1747, 4405, 4480, 4282, 9262, 6219, 3960, 4999, 1495, 6007, 9642, 3902, 3133, 1085, 3278, 1104, 5939, 7153, 971, 8733, 3785, 9056, 2020, 7249, 5021, 3384, 8740, 4593, 7869, 9941, 8813, 3688, 8139, 6436, 3742, 5503, 1587, 4766, 9846, 9117, 7001, 4853, 9346, 4927, 8480, 5298, 4753, 1151, 9768, 5405, 6196, 5721, 3419, 8090, 8166, 7834, 1480, 1150, 9002, 1134, 2237, 3995, 2029, 5336, 7050, 6857, 8794, 1754, 1184, 3558, 658, 6804, 8750, 5088, 1136, 626, 8462, 5203, 3196, 979, 7419, 1162, 5451, 6492, 1562, 8145, 8937, 8764, 4174, 7639, 8902, 7003, 765, 1554, 6135, 1689, 9530, 1398, 2273, 7925, 5948, 1036, 868, 4617, 1203, 7680, 7, 93, 3128, 5694, 6979, 7136, 8084, 5770, 9301, 1599, 737, 7018, 3774, 9843, 2296, 2287, 9875, 2349, 2469, 8941, 4973, 3798, 54, 2938, 4665, 3942, 3951, 9400, 3094, 2248, 3376, 1926, 5180, 1773, 3681, 1808, 350, 6669, 826, 539, 5313, 6193, 5752, 9370, 2782, 8399, 4881, 3166, 4906, 5829, 4827, 29, 6899, 9012, 6986, 4175, 1035, 8320, 7802, 3777, 6340, 7798, 7705]



def GP_pred(X, GP_model):
    X = X.reshape([1,X_feature.shape[1]])
    
    mean, var = GP_model.predict(X)[0][0][0], GP_model.predict(X)[1][0][0]
    return mean, np.sqrt(var)
    

# expected improvement
def EI(X, GP_model, y_best):
    xi = 0.01  ## was 0
#     can also use 0.01
    
    mean, std = GP_pred(X, GP_model)

    z = (y_best - mean - xi)/std
    return (y_best - mean - xi) * norm.cdf(z) + std * norm.pdf(z)

# lower confidence bound
def LCB(X, GP_model, ratio):
    
    mean, std = GP_pred(X, GP_model)
    
    return - mean + ratio * std

# probability of improvement
def PI(X, GP_model, y_best):
    xi = 0.01
#     can also use 0.01
    mean, std = GP_pred(X, GP_model)
    
    z = (y_best - mean - xi)/std
    return norm.cdf(z)


# if use isotropic kernels, ARD_ = False
# if use anisotropic kernels, ARD_ = True

ARD_ = True

Bias_kernel = GPy.kern.Bias(X_feature.shape[1], variance=1.)

Matern52_kernel = GPy.kern.Matern52(X_feature.shape[1], variance=1., ARD=ARD_) + Bias_kernel
Matern32_kernel = GPy.kern.Matern32(X_feature.shape[1], variance=1., ARD=ARD_) + Bias_kernel
Matern12_kernel = GPy.kern.Exponential(X_feature.shape[1], variance=1., ARD=ARD_) + Bias_kernel
RBF_kernel = GPy.kern.RBF(X_feature.shape[1], variance=1., ARD=ARD_) + Bias_kernel
MLP_kernel = GPy.kern.MLP(X_feature.shape[1], variance=1., ARD=ARD_) + Bias_kernel



# framework


# good practice to keep check of time used
start_time = time.time()

# these will carry results along optimization sequence from all n_ensemble runs
index_collection = []
X_collection = []
y_collection = []
TopCount_collection = []


pbar = tqdm(total=n_ensemble)

for s in tqdm(seed_list, desc='Optimization based on random seeds'):
    
    if len(index_collection) == n_ensemble:
        break
    
    print('initializing seed = ' +str(seed_list.index(s)))
    random.seed(s)
    
    indices = list(np.arange(N))
# index_learn is the pool of candidates to be examined
    index_learn = indices.copy()
# index_ is the list of candidates we have already observed
#     adding in the initial experiments
    index_ = random.sample(index_learn, n_initial)
    
#     list to store all observed good candidates' input feature X
    X_ = []
#     list to store all observed good candidates' objective value y
    y_ = []
#     number of top candidates found so far
    c = 0
#     list of cumulative number of top candidates found at each learning cycle
    TopCount_ = []
#     add the first n_initial experiments to collection
    for i in index_:
        X_.append(X_feature[i])
        y_.append(y[i])
        if i in top_indices:
            c += 1
        TopCount_.append(c)
        index_learn.remove(i)
           

#     for each of the the rest of (N - n_initial) learning cycles
#     this for loop ends when all candidates in pool are observed 
    for i in np.arange(len(index_learn)):
        
        y_best = np.min(y_)
        
        s_scaler = preprocessing.StandardScaler()
        X_train = s_scaler.fit_transform(X_)
        y_train = s_scaler.fit_transform([[i] for i in y_])
        
        try:
#             #TODO: select kernel for GP surrogate model
            GP_learn = GPy.models.GPRegression(X = X_train, 
                                               Y = y_train, 
                                               kernel= Matern52_kernel,
                                               noise_var = 0.01
                                              )

            GP_learn.optimize_restarts(num_restarts=10,
                                       parallel = True,
                                       robust = True,
                                       optimizer = 'bfgs',
                                       max_iters=100,
                                       verbose = False)
        except:
            break
        
#         by evaluating acquisition function values at candidates remaining in pool
#         we choose candidate with larger acquisition function value to be observed next   
        next_index = None
        max_ac = -10**10
        for j in index_learn:
            X_j = X_feature[j]
            y_j = y[j]
#             #TODO: select Acquisiton Function for BO
            ac_value = EI(X_j, GP_learn, y_best)
            #ac_value = LCB(X_j, GP_learn, 2)
            #ac_value = PI(X_j, GP_learn, y_best)
            
            if max_ac <= ac_value:
                max_ac = ac_value
                next_index = j
                
        
                
        X_.append(X_feature[next_index])
        y_.append(y[next_index])
        
        
        if next_index in top_indices:
            c += 1
        
        TopCount_.append(c)
        
        index_learn.remove(next_index)
        index_.append(next_index)        


    assert len(index_) == N
    
    index_collection.append(index_)
    X_collection.append(X_)
    y_collection.append(y_)
    TopCount_collection.append(TopCount_)
    
    pbar.update(1)
    #print('Finished seed')

pbar.close()
total_time = time.time() - start_time

master = np.array([index_collection, X_collection, y_collection, TopCount_collection, total_time])
#  #TODO: name output file
np.save(f'test_run_gp_surrogate_ARD_{dataset_name}_EI_xi_0.01.npy', master)
   

