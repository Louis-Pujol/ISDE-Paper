import numpy as np
from isde.estimators import GaussianKDE
from sklearn.model_selection import train_test_split
import json

def complete_graph_weigths(X_train, h1=0.1, h2=0.1, grid_size=20, held_out=False, X_test=None, bw_bysubsets=False,
                          by_subsets=None):
    
    _, d = X_train.shape
    
    partition = [ [i] for i in range(d) ]
    partition += [ [j, i] for i in range(d) for j in range(i)  ]
    
    
    if bw_bysubsets:
        
        weights = {}
        
        for a in [[j, i] for i in range(d) for j in range(i)]:
            weights[str(a)] = by_subsets[tuple(a)]["log_likelihood"]
            weights[str(a)]  -= by_subsets[(a[0], )]["log_likelihood"]
            weights[str(a)]  -= by_subsets[(a[1], )]["log_likelihood"]
            
        return weights
            
            
    
    grid_1d = np.linspace(0, 1, grid_size)
    grid_2d = np.array([[i, j] for i in grid_1d for j in grid_1d])

    pn1 = {} ##
    if held_out:
        pn2 = {}
        
    for p in partition:
        loc_d = len(p)
        if loc_d == 1:
            loc_grid = np.linspace(0, 1, grid_size).reshape(grid_size, 1)
            if not bw_bysubsets:
                h = h1
            else:
                h = by_subsets[tuple(p)]['params']['bandwidth']
        else:
            loc_grid = grid_2d
            if not bw_bysubsets:
                h = np.array([h1, h2])
            else:
                 h = by_subsets[tuple(p)]['params']['bandwidth']
            
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

def logdensity_from_tree(X, X_eval, graph, h1=.01, h2=.01,
                        bw_bysubsets=False, by_subsets=None):
    N, d = X.shape
    ll = 0
    
    for i in range(d):
        if not bw_bysubsets:
            ll += GaussianKDE(bandwidth=h1).score_samples(eval_points=X_eval[:, [i]], grid_points=X[:, [i]])
        else:
            loc_h = by_subsets[(i, )]['params']['bandwidth']
            ll += GaussianKDE(bandwidth=loc_h).score_samples(eval_points=X_eval[:, [i]], grid_points=X[:, [i]])
    
    for e in graph:
        if not bw_bysubsets:
            ll += GaussianKDE(bandwidth=h2).score_samples(eval_points=X_eval[:, e], grid_points=X[:, e])
            ll -= GaussianKDE(bandwidth=h1).score_samples(eval_points=X_eval[:, [e[0]]], grid_points=X[:, [e[0]]])
            ll -= GaussianKDE(bandwidth=h1).score_samples(eval_points=X_eval[:, [e[1]]], grid_points=X[:, [e[1]]])
        else:
            loc_h2d = by_subsets[(e[0], e[1])]['params']['bandwidth']
            loc_he0 = by_subsets[(e[0], )]['params']['bandwidth']
            loc_he1 = by_subsets[(e[1], )]['params']['bandwidth']
            ll += GaussianKDE(bandwidth=loc_h2d).score_samples(eval_points=X_eval[:, e], grid_points=X[:, e])
            ll -= GaussianKDE(bandwidth=loc_he0).score_samples(eval_points=X_eval[:, [e[0]]], grid_points=X[:, [e[0]]])
            ll -= GaussianKDE(bandwidth=loc_he1).score_samples(eval_points=X_eval[:, [e[1]]], grid_points=X[:, [e[1]]])
        
        
    return ll


def FDE(X, h=.01, bw_bysubsets=False, by_subsets=None):
    
    #LLW
    X_train, X_eval = train_test_split(X, test_size=0.33)
    weights = complete_graph_weigths(X_train, h1=h, h2=h, grid_size=200, held_out=True, X_test=X_eval,
                                     bw_bysubsets=bw_bysubsets, by_subsets=by_subsets)
    graph = kruskal(weights, threshold=0)[0]
    
    
    return graph