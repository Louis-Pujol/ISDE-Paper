#ISDE
import numpy as np
import itertools

from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, LpMinimize
from pulp import GLPK
from  pulp.apis import PULP_CBC_CMD

from sklearn.model_selection import train_test_split

import itertools
from ast import literal_eval

from pykeops.numpy import Genred

    


def ISDE(X, m, n, k, multidimensional_estimator, do_optimization=True, verbose=False, **params_estimator):
    '''
    
    Inputs
    - X : input dataset (numpy array)
    - m : size of set use to 
    - n : size of set used ton compute log-likelihoods
    - k : desired size of biggest bloc in the outputed partition
    '''
    
    by_subsets = {}
    
    N, d = X.shape
    W, Z = train_test_split(X, train_size=m, test_size=n)
    
    for i in range(1, k+1):
        if verbose:
            print("Computing estimators for subsets of size {}...".format(i))
            
        for S in itertools.combinations(range(d), i):
        
            f, f_params = multidimensional_estimator(W[:, S], params_estimator)
            ll = np.ma.masked_invalid(f.score_samples(grid_points = W[:, S], eval_points = Z[:, S])).mean()
            
            by_subsets[S] = {'log_likelihood': ll, 'params': f_params}
            
    if do_optimization:
        optimal_partition = find_optimal_partition(by_subsets, max_size=k, min_size=1, exclude = [], sense='maximize')[0]
        optimal_parameters = []
        
        for S in optimal_partition:
            optimal_parameters.append(by_subsets[tuple(S)]["params"])
		    
        return by_subsets, optimal_partition, optimal_parameters 
    else:

        return by_subsets




def find_optimal_partition(scores_by_subsets, max_size, min_size=1, exclude = [], sense='maximize'):
    
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
        
    
    ### exclude
    xs_name = [ list(literal_eval(i.name)) for i in xs]
    for p_exclude in exclude:
        model += lpSum( [xs[xs_name.index(s)] for s in p_exclude]) <= len(p_exclude) - 1 
    
    #Solve
    model.solve(PULP_CBC_CMD(msg=False))
    #if verbose != 0:
        #print("Status: {}, {}".format(model.status, LpStatus[model.status]))
        #print("Objective value : {}".format(model.objective.value()))
        
    output_dict = {var.name : var.value() for var in model.variables()  }
    out_partition = []
    for o in output_dict.keys():
        if output_dict[o] != 0:
            out_partition.append(list(literal_eval(o.replace("_", " "))))
            
    #if verbose != 0:
        #print("Output : {}".format(out_partition))
    
    return out_partition, model.objective.value()

        
def logdensity_from_partition(X, X_eval, partition, parameters, estimator):
    
    M = len(X_eval)
    log_density = np.zeros(len(X_eval))

    for i, S in enumerate(partition):

        loc_param = parameters[i]
        f = estimator(**loc_param)
        log_density += f.score_samples(grid_points=X[:, S], eval_points=X_eval[:, S])

    return log_density

  
# Estimators


def gaussian_kde(grid_points, eval_points, h, backend='auto'):
    

    N, d = grid_points.shape
    
    my_conv = Genred('Exp(- SqNorm2(x - y))', ['x = Vi({})'.format(d), 'y = Vj({})'.format(d)],
                     reduction_op='Sum',axis=0)

    
    C = np.sqrt(0.5) / h
    a = my_conv(C * np.ascontiguousarray(grid_points)
                ,C * np.ascontiguousarray(eval_points),
                backend=backend).transpose()[0]
    
    return a / (N*(h ** d)*np.power(2*np.pi, d/2))


class GaussianKDE:
    
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth
    
    def score_samples(self, grid_points, eval_points):
        """ Return log-likelihood evaluation
        """
        _, d = eval_points.shape
        return np.log( gaussian_kde(grid_points=grid_points, eval_points=eval_points, h=self.bandwidth) )
    
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
            scores[h].append(np.mean (np.log( gaussian_kde(grid_points=W_train, eval_points=W_test, h=h))))
            
    mean_scores = [np.mean(scores[h]) for h in hs]
    h_opt = hs[np.argmax(mean_scores)]
    
    kde = GaussianKDE(bandwidth=h_opt)
        
    return kde, {'bandwidth' : h_opt}

def Hold_out_KDE(W,params):
    
    hs = params['hs']
    n_train = params['n_train']
    
    W_train = W[0:n_train, :] 
    W_test = W[n_train::, ]
    
    scores = {h : [] for h in hs}
    for h in hs:
        tmp = np.log(gaussian_kde(grid_points=W_train, eval_points=W_test, h=h))
        scores[h].append(np.ma.masked_invalid(tmp).mean())
    
    
    h_opt = hs[np.argmax([scores[h] for h in hs])]
    kde = GaussianKDE(bandwidth=h_opt)
        
    return kde, {'bandwidth' : h_opt}

    
def KDE_fixed_h(W, params):
    
    h = params['h']
    
    if h == 'scott':
        n, d = W.shape
        h = n ** (- 1. / (d + 4))
    
    kde = GaussianKDE(bandwidth=h)
        
    return kde, {'bandwidth' : h}