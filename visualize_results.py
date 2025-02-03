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
import ternary
import pickle
import datetime
from collections import Counter
import matplotlib.ticker as ticker
import pyDOE
import random
from sklearn import preprocessing
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
import matplotlib.font_manager as font_manager
import copy
from scipy.interpolate import splrep
from scipy.interpolate import interp1d

# go to directory where datasets reside
# load a dataset
# dataset names = ['Crossed barrel', 'Perovskite', 'AgNP', 'P3HT', 'AutoAM']
dataset_name = 'P3HT'
raw_dataset = pd.read_csv('datasets/' + dataset_name + '_dataset.csv')
feature_name = list(raw_dataset.columns)[:-1]
objective_name = list(raw_dataset.columns)[-1]

ds = copy.deepcopy(raw_dataset) 
# only P3HT/CNT, Crossed barrel, AutoAM need this line; Perovskite and AgNP do not need this line.
ds[objective_name] = -raw_dataset[objective_name].values

ds_grouped = ds.groupby(feature_name)[objective_name].agg(lambda x: x.unique().mean())
ds_grouped = (ds_grouped.to_frame()).reset_index()

# pool size
# total number of data in set
N = len(ds_grouped)
print(N)
# number of top candidates, currently using top 5% of total dataset size
n_top = int(math.ceil(N * 0.05))
print(n_top)
# the top candidates and their indicies
top_indices = list(ds_grouped.sort_values(objective_name).head(n_top).index)

# Create list of files to be loaded
files = [
    f'test_run_gp_surrogate_ARD_{dataset_name}_LCB_ratio_2.npy',
    f'test_run_gp_surrogate_ARD_{dataset_name}_EI_xi_0.01.npy', 
    f'test_run_gp_surrogate_ARD_{dataset_name}_PI_xi_0.01.npy',
    f'test_run_rf_surrogate_{dataset_name}_LCB_ratio_2.npy',
    f'test_run_rf_surrogate_{dataset_name}_EI_xi_0.01.npy', 
    f'test_run_rf_surrogate_{dataset_name}_PI_xi_0.01.npy',
    f'test_run_gp_surrogate_{dataset_name}_LCB_ratio_2.npy',
    f'test_run_gp_surrogate_{dataset_name}_EI_xi_0.01.npy', 
    f'test_run_gp_surrogate_{dataset_name}_PI_xi_0.01.npy'
    ]

# Create list of labels corresponding to each results file
labels = [
    'GP M52 ARD: LCB$_{\overline{2}}$', 
    'GP M52 ARD: EI, \u03BE=0.01', 
    'GP M52 ARD: PI, \u03BE=0.01',
    'RF : LCB$_{\overline{2}}$', 
    'RF : EI, \u03BE=0.01', 
    'RF : PI, \u03BE=0.01',
    'GP M52 : LCB$_{\overline{2}}$', 
    'GP M52 : EI, \u03BE=0.01', 
    'GP M52 : PI, \u03BE=0.01',
    ]

# Create list of colors corresponding to each results file
colors = [
    'sienna',
    'orange',
    'gold',
    'darkblue', 
    'dodgerblue', 
    'lightblue', 
    '#006d2c', 
    'mediumseagreen', 
    'mediumspringgreen'
    ]


def P_rand(nn):
    x_random = np.arange(nn)
    
    M = n_top
    N = nn
    
    P = np.array([None for i in x_random])
    E = np.array([None for i in x_random])
    A = np.array([None for i in x_random])
    cA = np.array([None for i in x_random])
    
    P[0] = M / N
    E[0] = M / N
    A[0] = M / N
    cA[0] = A[0]
    

    for i in x_random[1:]:
        P[i] = (M - E[i-1]) / (N - i)
        E[i] = np.sum(P[:(i+1)])
        j = 0
        A_i = P[i]
        while j < i:
            A_i *= (1 - P[j])
            j+=1
        A[i] = A_i
        cA[i] = np.sum(A[:(i+1)])
        
    return E / M, cA


# List of seeds to create 100 (sets) X 5 (folds) = 500 (total sets) of 10 randomly selected runs (out of 50)
seed_list = [5782, 5776, 9975, 4569, 8020, 363, 9656, 992, 348, 6048, 4114, 7476, 4892, 9710, 9854, 5243, 
             2906, 5963, 3035, 5122, 9758, 4327, 4921, 6179, 1718, 441, 9326, 2153, 5079, 8192, 3646, 4413, 
             3910, 5370, 3070, 7130, 1589, 1668, 9842, 5275, 5468, 3677, 7183, 2773, 1309, 5516, 3572, 9312, 
             7390, 4433, 3686, 1981, 555, 8677, 3126, 5163, 9418, 3007, 4564, 5572, 1401, 5657, 9658, 2124, 
             6902, 4783, 8493, 4442, 7613, 5674, 6830, 4757, 6877, 9311, 6709, 582, 6770, 2555, 3269, 76, 
             7820, 8358, 7116, 9156, 3638, 529, 7482, 8503, 4735, 8910, 5588, 3726, 1115, 9644, 4702, 1966, 
             4006, 738, 575, 8393]


def aggregation_(seed, n_runs, n_fold):
    
    assert math.fmod(n_runs, n_fold) == 0
    fold_size = int(n_runs / n_fold)
    
    random.seed(seed)
    
    index_runs = list(np.arange(n_runs))
    
    agg_list = []
    
    i = 0
    
    # Size of folds (10) = runs based on random seeds (50) / number of folds (5)
    while i < n_fold:
    
        index_i = random.sample(index_runs, fold_size)
        for j in index_i:
            index_runs.remove(j)
            
        agg_list.append(index_i)
        
        i += 1
#     print(agg_list)    
    return agg_list


def avg_(x):
#     nsteps
    n_eval = len(x[0]) 
    
#     fold
    n_fold = 5
    
#     rows = # of ensembles = 50
    n_runs = len(x)
    
    assert math.fmod(n_runs, n_fold) == 0
    fold_size = int(n_runs / n_fold)
    
#     # of seeds 
    n_sets = len(seed_list)
    
    l_index_list = []
    
    # Get sets(100) X folds (5) = 500 total sets of seeds corresponding to runs (10)
    for i in np.arange(n_sets):
        
        s = aggregation_(seed_list[i], n_runs, n_fold)
        l_index_list.extend(s)

#     rows in l_index_list

    assert len(l_index_list) == n_sets * n_fold

    
    l_avg_runs = []

    # For each of the total sets average the runs selected based on the seeds of one set (10)
    for i in np.arange(len(l_index_list)):
        
        avg_run = np.zeros(n_eval)
        # Aggregation of runs that are randomly selected 
        for j in l_index_list[i]:
            
            avg_run += np.array(x[j])  # x contains all runs (50)
        #Average between the ensemble of runs (10 runs corresponding to 10 different random seeds)
        avg_run = avg_run/fold_size
        l_avg_runs.append(avg_run)  # in the end 500 total sets of averaged runs are created (all sets are based on the same 50 runs which are randomized)
    

    assert n_eval == len(l_avg_runs[0])
    assert n_sets * n_fold == len(l_avg_runs)
    
    mean_ = [None for i in np.arange(n_eval)]
    std_ = [None for i in np.arange(n_eval)]
    median_ = [None for i in np.arange(n_eval)]
    low_q = [None for i in np.arange(n_eval)]
    high_q = [None for i in np.arange(n_eval)]
    

#     5th, 95th percentile, mean, median are all accessible
    for i in np.arange(len(l_avg_runs[0])):  # for each of the 600 datapoints of a set
        i_column = []
        for j in np.arange(len(l_avg_runs)):   # for each of the sets
            i_column.append(l_avg_runs[j][i])  # extracting the values of all 500 sets for the each of the 600 datapoints
        
        # Compute statistics for every one of the datapoints
        i_column = np.array(i_column)
        mean_[i] = np.mean(i_column)
        median_[i] = np.median(i_column)
        std_[i] = np.std(i_column)
        low_q[i] = np.quantile(i_column, 0.05, out=None, overwrite_input=False, interpolation='linear')
        high_q[i] = np.quantile(i_column, 0.95, out=None, overwrite_input=False, interpolation='linear')
    
    return np.array(median_), np.array(low_q), np.array(high_q), np.array(mean_), np.array(std_)


def TopPercent(x_top_count, n_top, N):
    
    x_ = [[] for i in np.arange(len(x_top_count))]
    
    for i in np.arange(len(x_top_count)):
        for j in np.arange(N):
            if j < len(x_top_count[i]):
                x_[i].append(x_top_count[i][j] / n_top)
            else:
                x_[i].append(1)  # if all top candidates are discovered before the final datapoint then fill in with ones after the timestep the former is achieved

    return x_  #containes the top % discored at each iteration over all the best available candidates


# Initialize figure
fig = plt.figure(figsize=(12,12))
ax0 = fig.add_subplot(111)


# Iterate over the files and plot the corresponding results after aggregation
for file_i, file in enumerate(files):

    # Load ensemble calculation results from framework
    # for 50 ensembles, they take some time to run
    results = np.load(file, allow_pickle = True)

    # Aggregating the performance
    results = avg_(TopPercent(results[3], n_top, N))  # statistics for each of the iterations (600 iteration, i.e., as many as the datapoints)

    if file_i == 0:
        ax0.plot(np.arange(N)+1, P_rand(N)[0],'--',color='black',label='random baseline', linewidth=3.5)

    ax0.plot(np.arange(N) + 1, np.round(results[0].astype(np.double) / 0.005, 0) * 0.005, label = labels[file_i], color = colors[file_i], linewidth=3)
    ax0.fill_between(np.arange(N) + 1, np.round(results[1].astype(np.double) / 0.005, 0) * 0.005, np.round(results[2].astype(np.double) / 0.005, 0) * 0.005, color = colors[file_i], alpha=0.2)


# the rest are for visualization purposes, please adjust for different needs
font = font_manager.FontProperties(family='Arial', size = 26, style='normal')
leg = ax0.legend(prop = font, borderaxespad = 0,  labelspacing = 0.3, handlelength = 1.2, handletextpad = 0.3, frameon=False, loc = (0, 0.3))
for line in leg.get_lines():
    line.set_linewidth(4)
ax0.set_ylabel("Top%", fontname="Arial", fontsize=30, rotation='vertical')    
plt.hlines(0.8, 0, len(results[3]), colors='k', linestyles='--', alpha = 0.2)
ax0.set_ylim([0, 1.05])
ax0.set_xscale('log')
ax0.set_xlabel('learning cycle $i$', fontsize=30, fontname = 'Arial')
ax0.xaxis.set_tick_params(labelsize=30)
ax0.yaxis.set_tick_params(labelsize=30)
ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
plt.xticks([1, 2, 10, 100], ['1', '2','10', '10$^{\mathrm{2}}$'],fontname = 'Arial')  #'6$×$10$^{\mathrm{2}}$'
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontname = 'Arial')
plt.show()
fig.tight_layout()
print()