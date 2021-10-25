from sklearn.covariance import empirical_covariance
from ast import literal_eval
from scipy.stats import multivariate_normal
import numpy as np

def Sigma(struct, sigma):
    #Matrix abd structure as numpy array with values for alpha and sigma
    
    d = np.sum(struct)
    M = np.zeros(shape=(d, d))
    
    a = 0
    for i, s in enumerate(np.cumsum(struct)):
        b = s
        M[a:b, a:b] = sigma * np.ones(shape=(b - a, b - a))
        a = b
    
    np.fill_diagonal(M, 1)
    return M

def dimension(partition):
    out = 0
    for B in partition:
        p = len(B)
        out += p * (p-1) / 2
    
    return out


def bloc_diagonal_strucure(adjency_matrix):
    #Compute the bloc diagonal structure of an adgency matrix
    
    def merge(partition, i, j):
        for a, bloc in enumerate(partition):
            if i in bloc:
                bloc_i = bloc
            if j in bloc:
                bloc_j = bloc
        if bloc_i == bloc_j:
            out = partition
        else:
            out = [ bloc for bloc in partition if bloc not in [bloc_i, bloc_j] ] + [bloc_i + bloc_j]  
        
        #Arrange out
        out = [list(np.sort(o)) for o in out]
        order = np.argsort([o[0] for o in out])
        out = [out[order[i]] for i in range(len(out))]
        return out
    
    d = len(adjency_matrix)
    partition = [[i] for i in range(d)]
    for i, row in enumerate(adjency_matrix):
        for j, val in enumerate(row):
            if val == 1:
                partition = merge(partition, i, j)
    
    return partition

def compute_admissible_partitions(S):
    # Compute admissible partitions as defined in the paper
    # Input : S empirical covariance
    # Output : dictionnary whoses keys are break points and values associated partitions
    
    values = np.sort(np.unique(np.abs(S)))
    partitions = []

    for l in values:
        A = np.array(np.abs(S) > l, dtype=np.int)
        partitions.append(bloc_diagonal_strucure(A))
        
    out = {str(partitions[0]): {"lambda": 0}}

    for i in range(1, len(partitions)):
        if partitions[i] != partitions[i-1]:
            out[str(partitions[i])] = {"lambda": values[i]}
            
    return out

def KL(Sigma1, Sigma2):
    
    Prec1 = np.linalg.inv(Sigma1)
    Prec2 = np.linalg.inv(Sigma2)
    
    B = np.dot(Prec2 - Prec1, Sigma1)

    v = np.linalg.eig(B)[0]
    return np.sum(v - np.log(1 + v)) / 2

def empirical_covariance_partition(X, partition):
    _, d = X.shape
    emp_cov = np.zeros(shape=(d, d))
    for p in partition:
        loc_cov = empirical_covariance(X[:, p])
        for i in range(len(p)):
            for j in range(len(p)):
                emp_cov[p[i], p[j]] = loc_cov[i, j]
                
    return emp_cov

class BDCS:
    
    
    def __init__(self):
        a = 1
        
    def fit(self, X):
        S = empirical_covariance(X, assume_centered=True)
        d = S.shape[0]
        N = len(X)
        self.result = compute_admissible_partitions(S)
        # Compute covariances
        for partition in self.result.keys():
            partition = literal_eval(partition)
            
            emp_cov = np.zeros(shape=S.shape)
            for p in partition:
                loc_cov = empirical_covariance(X[:, p])
                for i in range(len(p)):
                    for j in range(len(p)):
                        emp_cov[p[i], p[j]] = loc_cov[i, j]
                self.result[str(partition)]['Covariance'] = emp_cov
                
        # Compute log_likelihoods
        for partition in self.result.keys():
            S = self.result[partition]['Covariance']
            self.result[partition]['log_Likelihood'] = np.mean(np.log(multivariate_normal(mean=np.zeros(d), cov=S).pdf(X)))
        
        #Compute dimensions
        for partition in self.result.keys():
            self.result[partition]['dimension'] = dimension(literal_eval(partition))
        
        #Compute kappa
        best = np.inf

        kappas = np.linspace(0, 10, 1000)
        best_dims = []
        for kappa in kappas:

            best = np.inf

            for i, partition in enumerate(self.result.keys()):

                loc_score = - self.result[partition]['log_Likelihood'] + kappa * (self.result[partition]['dimension'] / N )
                if loc_score < best:
                    best = loc_score
                    best_part = literal_eval(partition)
            best_dims.append(self.result[str(best_part)]['dimension'])
            
        kappa_star =  2 * kappas[np.argmax(np.abs(np.diff(best_dims)))]
        
        #Compute best partition
        best = np.inf

        for i, partition in enumerate(self.result.keys()):

            loc_score = - self.result[partition]['log_Likelihood'] +  kappa_star * (self.result[partition]['dimension'] / N )
            if loc_score < best:
                best = loc_score
                best_part = partition
                
        return self.result[best_part]['Covariance'], literal_eval(best_part)
    
def KL(Sigma1, Sigma2):
    
    Prec1 = np.linalg.inv(Sigma1)
    Prec2 = np.linalg.inv(Sigma2)
    
    B = np.dot(Prec2 - Prec1, Sigma1)

    v = np.linalg.eig(B)[0]
    return np.sum(v - np.log(1 + v)) / 2
