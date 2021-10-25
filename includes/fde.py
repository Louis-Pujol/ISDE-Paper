import numpy as np
from pykeops.numpy import LazyTensor as LazyTensor_np
from sklearn.model_selection import train_test_split
import json


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
 
 
def complete_graph_weigths(X_train, h1, h2, grid_size=20, held_out=False, X_test=None):
    
    _, d = X_train.shape
    
    partition = [ [i] for i in range(d) ]
    partition += [ [j, i] for i in range(d) for j in range(i)  ]
    
    grid_1d = np.linspace(0, 1, grid_size)
    grid_2d = np.array([[i, j] for i in grid_1d for j in grid_1d])

    pn1 = {} ##
    if held_out:
        pn2 = {}
        
    for p in partition:
        loc_d = len(p)
        if loc_d == 1:
            loc_grid = np.linspace(0, 1, grid_size).reshape(grid_size, 1)
            h = h1
        else:
            loc_grid = grid_2d
            h = np.array([h1, h2])
            
        #ps.gaussian_kde(eval_points=loc_grid, grid_points=X_train[:, p], h=h * np.ones(1))
        
#         pn1[str(p)] = KernelDensity(X=X_train[:, p], bw=h).score(x=loc_grid, method=method, n_jobs=n_jobs, log=False)
        pn1[str(p)] = np.exp(GaussianKDE(bandwidth=h).score_samples(eval_points=loc_grid, grid_points=X_train[:, p]))
        if held_out:
            pn2[str(p)] = np.exp(GaussianKDE(bandwidth=h).score_samples(eval_points=loc_grid, grid_points=X_test[:, p]))
            
    
    weights = {} # \hat{I}
    for a in [ [j, i] for i in range(d) for j in range(i)  ]:
        
        
        if not held_out :
            
            pn1xy = pn1[str(a)]
            pn1x = np.repeat(pn1[str([a[0]])], grid_size)
            pn1y = np.tile(pn1[str([a[1]])], grid_size)
            
            weights[str(a)] = np.mean(pn1xy * np.log(pn1xy / (pn1x * pn1y)))
        
        else:
            
            pn1xy = pn1[str(a)]
            pn2xy = pn2[str(a)]
            pn1x = np.repeat(pn1[str([a[0]])], grid_size)
            pn1y = np.tile(pn1[str([a[1]])], grid_size)
            
            indices2xy = np.argwhere(pn2xy != 0)
            
            indices2xy = indices2xy.reshape(len(indices2xy))
            
            weights[str(a)] = np.mean(pn2xy[indices2xy] * np.log(pn1xy[indices2xy] / (pn1x[indices2xy] * pn1y[indices2xy])))
                
            
            
        
    return weights


def kruskal(weights, threshold = None):
    
    #Find the dimension (maximum value in weigths.keys() + 1)
    d = 0
    for i in weights.keys():
        d = np.max([d] + json.loads(i))
    d += 1
    
    cliques = [ [i] for i in range(d)] #At the beginning : every variable alone
    graph = [] # Will be the final graph (list of edges)
    graph_weights = []  # corresponding weights ( \hat{I} )
    
    sort_orders = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    
    if threshold != None :
        sort_orders2 = []
        for i in sort_orders:
            if i[1] > threshold:
                sort_orders2.append(i)
        sort_orders = sort_orders2
    
    
    for (pairs, s) in sort_orders:
        
        a, b = json.loads(pairs)[0], json.loads(pairs)[1]
        
                
        combine = True
        clique_a = []
        clique_b = []
            
        for c in cliques:
            if a in c and b in c: #Si l'arete ajoute un cycle
                combine = False
            if a in c:
                clique_a = c
            if b in c:
                clique_b = c
                
        if combine == True: 
            graph.append([a, b])
            graph_weights.append(s)
            #MàJ de cliques
            cliques = [c for c in cliques if c != clique_a and c != clique_b]
            cliques.append(clique_a + clique_b)
                        
                
    return(graph, graph_weights)

def logdensity_from_tree(X, X_eval, graph, h1, h2):
    N, d = X.shape
    ll = 0
    
    for i in range(d):
        ll += GaussianKDE(bandwidth=h1).score_samples(eval_points=X_eval[:, [i]], grid_points=X[:, [i]])
    
    for e in graph:
    
        ll += GaussianKDE(bandwidth=h2).score_samples(eval_points=X_eval[:, e], grid_points=X[:, e])
        ll -= GaussianKDE(bandwidth=h1).score_samples(eval_points=X_eval[:, [e[0]]], grid_points=X[:, [e[0]]])
        ll -= GaussianKDE(bandwidth=h1).score_samples(eval_points=X_eval[:, [e[1]]], grid_points=X[:, [e[1]]])
        
        
    return ll


def FDE(X):
    
    #LLW
    X_train, X_eval = train_test_split(X, test_size=0.33)
    weights = complete_graph_weigths(X_train, h1=0.01, h2=0.01, grid_size=200, held_out=True, X_test=X_eval)
    graph = kruskal(weights, threshold=0)[0]
    
    
    return graph

