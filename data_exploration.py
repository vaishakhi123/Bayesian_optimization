"""
Data exploratio.
"""

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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import matplotlib.font_manager as font_manager
import copy
from scipy.interpolate import splrep
from scipy.interpolate import interp1d
from sklearn.pipeline import make_pipeline
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import permutation_importance
import pickle

# go to directory where datasets reside
# load a dataset
dataset_names = ['Crossed barrel', 'Perovskite', 'AgNP', 'P3HT', 'AutoAM']


for dataset_i, dataset_name in enumerate(dataset_names):

    # Load dataset and extract features
    raw_dataset = pd.read_csv('datasets/' + dataset_name + '_dataset.csv')
    feature_name = list(raw_dataset.columns)[:-1]
    objective_name = list(raw_dataset.columns)[-1]

    # Plot pairplots
    g = sns.pairplot(raw_dataset[feature_name], corner=True, diag_kind='kde')
    g.map_lower(sns.kdeplot, levels=4, color=".2")
    plt.suptitle(f'{dataset_names[dataset_i]}')

    # Compute correlation matrices
    pearson = np.round(raw_dataset[feature_name].corr(method='pearson'), 2)
    spearman = np.round(raw_dataset[feature_name].corr(method='spearman'), 2)

    # Plot correlation matrices
    fig, axes = plt.subplots(1,2,figsize=(12,12))
    plt.suptitle(f'{dataset_names[dataset_i]}')
    plt.sca(axes[0])
    plt.title(f'Pearson correlation')
    sns.heatmap(pearson, annot=True)
    plt.sca(axes[1])
    plt.title(f'Spearman correlation')
    sns.heatmap(pearson, annot=True)

    # Scale data
    X = pd.DataFrame(raw_dataset, columns=feature_name)
    y = raw_dataset[objective_name]
    s_scaler = preprocessing.StandardScaler()
    X = pd.DataFrame(s_scaler.fit_transform(X), columns=feature_name)
    y = s_scaler.fit_transform([[i] for i in y]).reshape(-1,)

    # Instantiate estimators
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    gp = GaussianProcessRegressor(Matern(length_scale=1, nu=2.5), n_restarts_optimizer=10, random_state=117)
    gp_ARD = GaussianProcessRegressor(Matern(length_scale=np.zeros(len(feature_name)), nu=2.5), n_restarts_optimizer=10, random_state=117)
    Bias_kernel = GPy.kern.Bias(X.shape[1], variance=1.)
    Matern52_kernel = GPy.kern.Matern52(X.shape[1], variance=1., ARD=False) + Bias_kernel
    Matern52_kernel_ARD = GPy.kern.Matern52(X.shape[1], variance=1., ARD=True) + Bias_kernel

    # Train estimators
    # Train Random Forest
    rf.fit(X, y)
    #gp.fit(X, y)
    #gp_ARD.fit(X, y)

    # Plot partial dependence
    fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
    plt.suptitle(f'{dataset_names[dataset_i]}')
    tree_disp = PartialDependenceDisplay.from_estimator(rf, X, features=feature_name, ax=ax, line_kw={"label": "Random Forest", 'color': 'blue'})
    axes = tree_disp.axes_.reshape(-1,)[:X.shape[1]]
    #gp_disp = PartialDependenceDisplay.from_estimator(gp, X, features=feature_name, line_kw={"label": "Isotropic GP",  'color':'green'}, ax=axes)
    #gp_ARD_disp = PartialDependenceDisplay.from_estimator(gp_ARD, X, features=feature_name, line_kw={"label": 'Anisotropic GP', "color": "red"}, ax=axes)

    # Plot importance based on impurity decrease
    importances = rf.feature_importances_
    std = np.std([rf.feature_importances_ for tree in rf.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_name)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    # Plot permutation importance    
    result = permutation_importance(
        rf, X, y, n_repeats=15, random_state=117, n_jobs=1
    )
    forest_importances = pd.Series(result.importances_mean, index=feature_name)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()

    # Get length scales from anisotropic sklearn GP
    #sklearn_gp_length_scale = gp.kernel_.get_params()['length_scale']
    #sklearn_gp_ARD_length_scale = gp_ARD.kernel_.get_params()['length_scale']

    # Train Gausiian Process with isotropic kernel
    y = y.reshape(-1,1)
    model_file = f'gp_{dataset_name}.pkl'
    if os.path.exists(model_file):
        with open(model_file, 'rb') as file:
            GP_learn = pickle.load(file)
    else:
        GP_learn = GPy.models.GPRegression(X = X,
                                            Y = y, 
                                            kernel= Matern52_kernel,
                                            noise_var = 0.01
                                            )
        GP_learn.optimize_restarts(num_restarts=10, optimizer = 'bfgs', robust = True, max_iters=100, verbose = True)
        
        # Save the fitted model
        with open(model_file, 'wb') as file:
            pickle.dump(GP_learn, file)

    # Train Gaussian Process with anisotropic kernel
    model_file = f'gp_ARD_{dataset_name}.pkl'
    if os.path.exists(model_file):
        with open(model_file, 'rb') as file:
            GP_learn_ARD = pickle.load(file)
    else:
        GP_learn_ARD = GPy.models.GPRegression(X = X, 
                                                Y = y, 
                                                kernel= Matern52_kernel_ARD,
                                                noise_var = 0.01
                                                )
        GP_learn_ARD.optimize_restarts(num_restarts=10, optimizer = 'bfgs', robust = True, max_iters=100, verbose = True)

        # Save the fitted model
        with open(model_file, 'wb') as file:
            pickle.dump(GP_learn_ARD, file)

    # Plot lengthscales
    fig, ax = plt.subplots()
    plt.bar(feature_name, 1/GP_learn_ARD.sum.Mat52.lengthscale, width=0.5)
    plt.xlabel('Features')
    plt.ylabel('Inverse Length Scale')
    plt.title(f'{dataset_name}')
    fig.tight_layout()
    
    print()
