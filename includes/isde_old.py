import numpy as np
import itertools

from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, LpMinimize
from pulp import GLPK

from sklearn.model_selection import train_test_split

import itertools
from ast import literal_eval


#from .estimators.GaussianKDE import *
#from .estimators.EmpiricalCovariance import *
    


def ISDE(X, m, n, multidimensional_estimator, **params_estimator):
    '''
    Run ISDE
    
    
    Inputs
    - X : input dataset (numpy array)
    - m : size of set used to evaluate multivariate estimators
    - n : size of set used to compute log-likelihoods
    - multidimensional_estimator : estimator to perform local density estimation, defined in this file : CVKDE, KDE_fixed_h, EmpiricalCovariance
    - **params_estimator : if applicable, parameters 
    '''
    
    by_subsets = {}
    
    N, d = X.shape
    W, Z = train_test_split(X, train_size=m, test_size=n)
    
    for i in range(1, d+1):
        for S in itertools.combinations(range(d), i):
        
            f, f_params = multidimensional_estimator(W[:, S], params_estimator)
            ll = np.mean( f.score_samples(grid_points = W[:, S], eval_points = Z[:, S]) )
            
            by_subsets[S] = {'log_likelihood': ll, 'params': f_params}
            
    optimal_partition = find_optimal_partition(by_subsets, max_size=d, min_size=1, exclude = [], sense='maximize')[0]
    
    optimal_parameters = []
    for S in optimal_partition:
        optimal_parameters.append(by_subsets[tuple(S)]["params"])
        
    return optimal_partition, optimal_parameters




def find_optimal_partition(scores_by_subsets, max_size, min_size=1, exclude = [], sense='maximize'):
    '''
    Routine to select partition based on linear programming formulation usind PulP package
    
    Inputs :
    - scores_by_subsets : dictionnary (indexed by subsets) of subdictionnaries with key "log_likelihood"
    - max_size : only keep the subsets of cardinal <= max_size to do the selection 
    - min_size : only keep the subsets of cardinal >= min_size to do the selection
    - exclude (list of partitions) : exclude some partitions of the selection
    - sense (str) (default "maximize") : solve the max ("maximize") or the min ("minimize") problem
    '''
    
    nb_to_exclude = len(exclude)
    
    ### Create Graph
    weights = {}
    edges = []
    vertices = []

    for s in scores_by_subsets.keys():
        
            for i in s:
                if i not in vertices:
                    vertices.append(i)

            if len(s) <= max_size and len(s) >= min_size:
                edges.append(s)
                weights[s] = scores_by_subsets[s]["log_likelihood"]
    
    ### Create model and variables
    if sense == 'maximize':
        model = LpProblem(name="Best_partition", sense=LpMaximize)
    elif sense == 'minimize':
        model = LpProblem(name="Best_partition", sense=LpMinimize)
    xs = []
    
    for e in edges:
        #Replace ' ' by '' to avoid extras '_'
        xs.append(LpVariable(name=str(e).replace(' ', ''), lowBound=0, upBound=1, cat="Integer"))
    
    ### Cost function
    objective = lpSum([weights[e] * xs[i] for (i, e) in enumerate(edges)])
    model += objective
    

    ### Constrains
    A = np.zeros(shape=(len(vertices), len(edges)))
    for (i, e) in enumerate(edges):
        for v in e:
            A[v, i] = 1
    
    for (i, e) in enumerate(vertices):
        model += (lpSum([A[i, j] * xs[j] for j in range(len(edges)) ]) == 1)
        
    
    ### If applicable: exclude
    if len(exclude) > 1:
    
        xs_name = [ list(literal_eval(i.name)) for i in xs]
        for p_exclude in exclude:
            model += lpSum( [xs[xs_name.index(s)] for s in p_exclude]) <= len(p_exclude) - 1 
    
    #Solve
    model.solve()
        
    output_dict = {var.name : var.value() for var in model.variables()  }
    out_partition = []
    for o in output_dict.keys():
        if output_dict[o] != 0:
            out_partition.append(list(literal_eval(o.replace("_", " "))))
            
    
    return out_partition, model.objective.value()

        
def logdensity_from_partition(X, X_eval, partition, parameters, estimator):
    '''
    Compute the log_density over a partition for a given familiy of estimators
    
    Inputs:
    - X (numpy array) : points used to calibratye hyperparameters
    - X_eval (numpy array) : points to evaluate density
    - partition (list of lists) : partition of the features
    - parameters : list of parameters for each set in the partition
    - estimator : estimator to perform local density estimation, defined in this file : CVKDE, KDE_fixed_h, EmpiricalCovariance
    '''
    M = len(X_eval)
    log_density = np.zeros(len(X_eval))

    for i, S in enumerate(partition):

        loc_param = parameters[i]
        f = estimator(**loc_param)
        log_density += f.score_samples(grid_points=X[:, S], eval_points=X_eval[:, S])

    return log_density


###### Subroutines

# EmpiricalCovariance
from sklearn.covariance import empirical_covariance
from scipy.stats import multivariate_normal

class Covariance:
    
    def __init__(self, cov):
        self.cov = cov
        
    def score_samples(self, grid_points, eval_points):
        """ Return log-likelihood evaluation
        """
        _, d = eval_points.shape
        var = multivariate_normal(mean=np.zeros(d), cov=self.cov)
        return np.log(var.pdf(eval_points))
    
def EmpCovariance(W, params):
    
    emp_cov = empirical_covariance(W, assume_centered=True)
    estimator = Covariance(cov=emp_cov)
    
    return estimator, {'covariance' : emp_cov}

# GaussianKDE
from pykeops.numpy import LazyTensor as LazyTensor_np

def gaussian_kde(grid_points, eval_points, h):
    """
    Perform Kernel Density estimation with grid_points and eval_points and a vector of bandwidths
    Works with gpu and torch (and cpu ?)
    
    Inputs : 
    - eval_points : (m, d) array, points to evaluate density
    - grid_points : (N, d), points to construct density estimator
    - h : d-dimensional array, directional bandwidths
    
    Output:
    - m-dimensional array, evaluation of density for points of eval_points
    """
    
    N, d = grid_points.shape
    x_i = LazyTensor_np(eval_points[:, None, :])  # (M, 1, d) KeOps LazyTensor, wrapped around the numpy array eval_points
    X_j = LazyTensor_np(grid_points[None, :, :])  # (1, N, d) KeOps LazyTensor, wrapped around the numpy array grid_points
    h_l = LazyTensor_np(h)

    D_ij = ( -0.5 * (((x_i - X_j) / h_l) ** 2).sum(-1))  # **Symbolic** (M, N) matrix of squared distances
    s_i = D_ij.exp().sum(dim=1).ravel()  # genuine (M,) array of integer indices
    
    out = s_i / (N*np.prod(h)*np.power(2*np.pi, d/2))
    
    return out


class GaussianKDE:
    
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth
    
    def score_samples(self, grid_points, eval_points):
        """ Return log-likelihood evaluation
        """
        _, d = eval_points.shape
        return np.log( gaussian_kde(grid_points=grid_points, eval_points=eval_points, h=self.bandwidth*np.ones(d)) )
    
def CVKDE(W, params):
    
    hs = params['hs']
    if 'n_fold' in params:
        n_fold = params['n_fold']
    else:
        n_fold = 5
    
    
    scores = {h : [] for h in hs}
    
    m, d = W.shape
    step = int(m / n_fold)
    indexes = list(range(m))

    for i in range(int(m/step)):
                            
        if i != int(m/step) - 1 :
            test_indexes = indexes[i * step: (i+1)*step]
            train_indexes = [i for i in indexes if i not in test_indexes]
                                
        else:
            test_indexes = indexes[i * step:]
            train_indexes = [i for i in indexes if i not in test_indexes]
                                
        W_train, W_test = W[train_indexes, :], W[test_indexes, :]
                            
        for h in hs:
            scores[h].append(np.mean (np.log( gaussian_kde(grid_points=W_train, eval_points=W_test, h=h*np.ones(d)))))
            
    mean_scores = [np.mean(scores[h]) for h in hs]
    h_opt = hs[np.argmax(mean_scores)]
    
    kde = GaussianKDE(bandwidth=h_opt)
        
    return kde, {'bandwidth' : h_opt}
    
def KDE_fixed_h(W, params):
    
    h = params['h']
    
    if h == 'scott':
        n, d = W.shape
        h = n ** (- 1. / (d + 4))
    
    kde = GaussianKDE(bandwidth=h)
        
    return kde, {'bandwidth' : h}





