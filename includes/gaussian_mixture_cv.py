#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 10:18:52 2022

@author: louis
"""

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
import numpy as np


def GaussianMixture_cv(X, min_components, max_components, criterion='cross_val'):
    """
    Fit a gaussian mixture model on X with a selection of the number of components and return
    the model.
    
    Inputs :
    - X : data
    - min_components (int) : minimum number of components
    - max_components (int) : maximum number of components
    - criterion ('BIC' or 'cross_val') : criterion for model selection
    
    Output :
    gm (sklearn.mixture.GaussianMixture) : model fitted on the data    
    """
    
    components = list(range(min_components, max_components+1))

    if criterion == 'BIC':
        BICs = []

        for n_c in components:
            gm = GaussianMixture(n_components=n_c, random_state=0).fit(X)
            BICs.append(gm.bic(X))

        #Select the right number of components (minimizing BIC criterion)
        selected_nc = components[np.argwhere(BICs==np.min(BICs))[0][0]]
        gm = GaussianMixture(n_components=selected_nc, random_state=0).fit(X)

    if criterion == 'cross_val':


        from sklearn.model_selection import GridSearchCV

        # use grid search cross-validation to optimize the bandwidth
        params = {"n_components": components}
        grid = GridSearchCV(GaussianMixture(random_state=0), params, n_jobs=-1)
        grid.fit(X)
        gm = grid.best_estimator_


    #Compute log-density on X_validation
    return gm