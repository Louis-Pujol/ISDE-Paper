import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
import includes.isde as isde



def partition_random(d, k_max, k_min=1):
    l = list(range(d))
    separators = [0]
    cumsum = 0
    while cumsum < d:
        a = np.random.randint(k_min, k_max+1)
        if cumsum + a > d:
            a = d - cumsum
            cumsum = d
        else:
            cumsum = cumsum + a
        separators.append(cumsum)
    np.random.shuffle(l)
    
    return  [ sorted(l[separators[i]:separators[i+1]]) for i in range(len(separators)-1) ]

def logdensity_from_partition(X_grid, X_eval, partition, by_subsets):
    logdensity_isde = np.zeros(X_eval.shape[0])
    for S in partition:
        loc_d = len(S)
        loc_h = by_subsets[tuple(S)]['params']['bandwidth']
        logdensity_isde += isde.GaussianKDE(bandwidth=loc_h).score_samples(grid_points=X_grid[:, S], eval_points=X_eval[:, S])

    return np.ma.masked_invalid(logdensity_isde)

def best_worst_rnd(X_train, X_validation, by_subsets, k, exp_name):

    how_many_partitions = 3

    # Find best partition, worst partitions and generate random partitions
    best_partitions = []
    for i in range(how_many_partitions):

        partition = isde.find_optimal_partition(scores_by_subsets=by_subsets, max_size=k,
                                                min_size=1, exclude = best_partitions, sense='maximize')[0]

        best_partitions.append(partition)

    worst_partitions = []
    for i in range(how_many_partitions):

        partition = isde.find_optimal_partition(scores_by_subsets=by_subsets, max_size=k,
                                                min_size=1, exclude = worst_partitions, sense='minimize')[0]

        worst_partitions.append(partition)

    _, d = X_train.shape
    rnd_partitions = []
    for i in range(how_many_partitions):

        partition = partition_random(d=d, k_min=1, k_max=k)
        rnd_partitions.append(partition)

    #Compute empirical log-likelihood for these
    M = 10
    N_v = 2000
    _, d = X_validation.shape

    lls_best = np.zeros(shape=(M, how_many_partitions))
    lls_worst = np.zeros(shape=(M, how_many_partitions))
    lls_rnd = np.zeros(shape=(M, how_many_partitions))
    lls_single = np.zeros(M)

    N_valid, d = X_validation.shape
    indices = list(range(N_valid))


    for i in range(M):
        np.random.shuffle(indices)
        X_v = X_validation[indices[0:N_v], :]

        p = [[i] for i in range(d)]
        logdensity_single = logdensity_from_partition(X_grid=X_train, X_eval=X_v, partition=p, by_subsets=by_subsets)
        lls_single[i] = logdensity_single.mean()

        for j, p in enumerate(best_partitions):

            logdensity_best = logdensity_from_partition(X_grid=X_train, X_eval=X_v, partition=p, by_subsets=by_subsets)
            lls_best[i, j] = logdensity_best.mean()


        for j, p in enumerate(worst_partitions):

            logdensity_worst = logdensity_from_partition(X_grid=X_train, X_eval=X_v, partition=p, by_subsets=by_subsets)
            lls_worst[i, j] = logdensity_worst.mean()

        for j, p in enumerate(rnd_partitions):

            logdensity_rnd = logdensity_from_partition(X_grid=X_train, X_eval=X_v, partition=p, by_subsets=by_subsets)
            lls_rnd[i, j] = logdensity_rnd.mean()

    #Plot

    mpl.rcParams['figure.dpi'] = 300


    df = pd.DataFrame()
    for i in range(how_many_partitions):
        df["Best" + str(i+1)] = lls_best[:, i] 

    for i in range(how_many_partitions):
        df["Rnd" + str(i+1)] = lls_rnd[:, i] 

    for i in range(how_many_partitions):
        df["Worst" + str(i+1)] = lls_worst[:, i]

    sns.set_style("whitegrid")
    ax = sns.boxplot(data=df)
    plt.ylabel("log-likelihood")
    plt.tight_layout()
    plt.show()
    
    sns.set_style("whitegrid")
    ax = sns.boxplot(data=df)
    plt.ylabel("log-likelihood")
    plt.tight_layout()
    plt.savefig("data/" + exp_name + "/scores_best_rnd_worst.png")
    plt.clf()

def edit(p1, p2):
    
    return 2 * sum([len(np.intersect1d(s1, s2)) != 0 for s1 in p1 for s2 in p2]) - len(p1) - len(p2)

def edit_distance_best_rnd_worst(best_partition, by_subsets, d, k, exp_name):

    how_many_partitions = 10

    # Find best partition, worst partitions and generate random partitions
    best_partitions = []
    for i in range(how_many_partitions):

        partition = isde.find_optimal_partition(scores_by_subsets=by_subsets, max_size=k,
                                                min_size=1, exclude = best_partitions, sense='maximize')[0]

        best_partitions.append(partition)

    worst_partitions = []
    for i in range(how_many_partitions):

        partition = isde.find_optimal_partition(scores_by_subsets=by_subsets, max_size=k,
                                                min_size=1, exclude = worst_partitions, sense='minimize')[0]

        worst_partitions.append(partition)

    rnd_partitions = []
    for i in range(how_many_partitions):

        partition = partition_random(d=d, k_min=1, k_max=k)
        rnd_partitions.append(partition)

    out = np.zeros(shape=(10, 3))

    for i in range(1, 11):
        out[i-1, 0] = edit(best_partition, best_partitions[i-1])

    print()
    for i in range(10):
        p = partition_random(d=d, k_min=1, k_max=k)
        out[i, 1] = edit(best_partition, p)

    print()

    for i in range(len(worst_partitions)):
        out[i, 2] = edit(best_partition, worst_partitions[i])

    mpl.rcParams['figure.dpi'] = 300
    sns.set_style("whitegrid")
    plt.scatter(np.zeros(10), out[:, 0])
    plt.scatter(np.ones(10), out[:, 1])
    plt.scatter(2 * np.ones(10), out[:, 2])
#     plt.scatter([3], [edit(best_partition, [[i] for i in range(d)])])
    plt.xticks(np.arange(3), ['10 best', '10 random', '10 worst'])
    plt.ylabel("Edit distance to partition outputted by ISDE")
    plt.tight_layout()
    plt.show()
    
    plt.scatter(np.zeros(10), out[:, 0])
    plt.scatter(np.ones(10), out[:, 1])
    plt.scatter(2 * np.ones(10), out[:, 2])
#     plt.scatter([3], [edit(best_partition, [[i] for i in range(d)])])
    plt.xticks(np.arange(3), ['10 best', '10 random', '10 worst'])
    plt.ylabel("Edit distance to partition outputted by ISDE")
    plt.tight_layout()
    plt.savefig("data/" + exp_name + "/edit_best_rnd_worst.png")
    plt.clf()

def edit_plus_one(partition, k):
    
    cut_or_merge = np.random.randint(2)
    lens = sorted([len(i) for i in partition])
    
    if max(lens) == 1:
        cut_or_merge = 1
       
    if lens[0] + lens[1] > k:
        cut_or_merge = 0
    

    if cut_or_merge == 0:
        a = [1]
        while len(a) == 1:
            a = partition[np.random.randint(len(partition))]

        where_cut = np.random.randint(1, len(a))
        part = [p.copy() for p in partition if p != a]
        part.append(a[0:where_cut])
        part.append(a[where_cut::])

    else:
        a = [1, 2, 3]
        b = [1, 2, 3]
        while a == b or len(a) + len(b) > k:
            a = partition[np.random.randint(len(partition))]
            b = partition[np.random.randint(len(partition))]
            
        part = [p.copy() for p in partition if p != a and p != b]
        merged = a.copy()
        for i in b:
            merged.append(i)
        part.append(sorted(merged))

    return part

def random_walk(X_train, X_validation, best_partition, by_subsets, k, exp_name):

    partitions_from_walk = []
    edit_distances = []

    n_walks = 5
    length_walks = 40


    for j in range(n_walks):
        a = best_partition.copy()

        for i in range(length_walks):
            a = edit_plus_one(a, k)
            partitions_from_walk.append(a)
            edit_distances.append(edit(best_partition, a))


    M = 10
    N_v = 2000

    lls = np.zeros(shape=(M, len(partitions_from_walk) + 1))

    N_valid, d = X_validation.shape
    indices = list(range(N_valid))


    for i in range(M):
        np.random.shuffle(indices)
        X_v = X_validation[indices[0:N_v], :]

        logdensity = logdensity_from_partition(X_grid=X_train, X_eval=X_v, partition=best_partition, by_subsets=by_subsets)
        lls[i, len(partitions_from_walk)] = logdensity.mean()

        for j, p in enumerate(partitions_from_walk):

            logdensity = logdensity_from_partition(X_grid=X_train, X_eval=X_v, partition=p, by_subsets=by_subsets)
            lls[i, j] = logdensity.mean()

    mpl.rcParams['figure.dpi'] = 300
    sns.set_style("whitegrid")
    means = np.ma.masked_invalid(lls).mean(axis=0)
    plt.xlabel("Edit distance from partition outputted by ISDE")
    plt.ylabel("Mean log-likelihood")
    plt.scatter(edit_distances + [0], means)
    plt.tight_layout()
    plt.show()
    
    
    plt.xlabel("Edit distance from partition outputted by ISDE")
    plt.ylabel("Mean log-likelihood")
    plt.scatter(edit_distances + [0], means)
    plt.tight_layout()
    plt.savefig("data/" + exp_name + "/log_likelihood_walks.png")
    plt.clf()