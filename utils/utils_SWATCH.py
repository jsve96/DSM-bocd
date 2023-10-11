
# -*- coding: utf-8 -*-

"""
Evaluation metrics

Author: G.J.J. van den Burg
Copyright (c) 2020 - The Alan Turing Institute
License: See the LICENSE file.

"""

import numpy as np
import json

def true_positives(T, X, margin=5):

    X = set(list(X))
    TP = set()
    for tau in T:
        close=[(abs(tau-x),x) for x in X if abs(tau-x)<=margin]
        close.sort()
        if not close:
            continue
        dist, xstar = close[0]
        TP.add(tau)
        X.remove(xstar)
    return TP

def f_measure(annotations, predictions, margin=5, alpha=0.5, return_PR=False):
    Tks = {k+1: set(annotations[uid]) for k, uid in enumerate(annotations)}
    for Tk in Tks.values():
        Tk.add(0)
    
    X = set(predictions)
    X.add(0)

    Tstar = set()
    for Tk in Tks.values():
        for tau in Tk:
            Tstar.add(tau)

    K = len(Tks)

    P = len(true_positives(Tstar,X, margin=margin)) / len(X)

    TPk = { k: true_positives(Tks[k],X, margin=margin) for k in Tks}
    R = 1/K*sum(len(TPk[k]) /len(Tks[k]) for k in Tks)

    F = P*R /(alpha * R +(1-alpha)*P)
    if return_PR:
        return F, P , R
    return F

def overlap(A, B):
    """Return the overlap (i.e. Jaccard index) of two sets

    >>> overlap({1, 2, 3}, set())
    0.0
    >>> overlap({1, 2, 3}, {2, 5})
    0.25
    >>> overlap(set(), {1, 2, 3})
    0.0
    >>> overlap({1, 2, 3}, {1, 2, 3})
    1.0
    """
    return len(A.intersection(B)) / len(A.union(B))


def partition_from_cps(locations, n_obs):
    """Return a list of sets that give a partition of the set [0, T-1], as
    defined by the change point locations.

    >>> partition_from_cps([], 5)
    [{0, 1, 2, 3, 4}]
    >>> partition_from_cps([3, 5], 8)
    [{0, 1, 2}, {3, 4}, {5, 6, 7}]
    >>> partition_from_cps([1,2,7], 8)
    [{0}, {1}, {2, 3, 4, 5, 6}, {7}]
    >>> partition_from_cps([0, 4], 6)
    [{0, 1, 2, 3}, {4, 5}]
    """
    T = n_obs
    partition = []
    current = set()

    all_cps = iter(sorted(set(locations)))
    cp = next(all_cps, None)
    for i in range(T):
        if i == cp:
            if current:
                partition.append(current)
            current = set()
            cp = next(all_cps, None)
        current.add(i)
    partition.append(current)
    return partition


def cover_single(S, Sprime):
    """Compute the covering of a segmentation S by a segmentation Sprime.

    This follows equation (8) in Arbaleaz, 2010.

    >>> cover_single([{1, 2, 3}, {4, 5, 6}], [{1, 2, 3}, {4, 5}, {6}])
    0.8333333333333334
    >>> cover_single([{1, 2, 3, 4, 5, 6}], [{1, 2, 3, 4}, {5, 6}])
    0.6666666666666666
    >>> cover_single([{1, 2, 3}, {4, 5, 6}], [{1, 2}, {3, 4}, {5, 6}])
    0.6666666666666666
    >>> cover_single([{1}, {2}, {3}, {4, 5, 6}], [{1, 2, 3, 4, 5, 6}])
    0.3333333333333333
    """
    T = sum(map(len, Sprime))
    assert T == sum(map(len, S))
    C = 0
    for R in S:
        C += len(R) * max(overlap(R, Rprime) for Rprime in Sprime)
    C /= T
    return C


def covering(annotations, predictions, n_obs):
    
    """Compute the average segmentation covering against the human annotations.

    annotations : dict from user_id to iterable of CP locations
    predictions : iterable of predicted Cp locations
    n_obs : number of observations in the series

    >>> covering({1: [10, 20], 2: [10], 3: [0, 5]}, [10, 20], 45)
    0.7962962962962963
    >>> covering({1: [], 2: [10], 3: [40]}, [10], 45)
    0.7954144620811286
    >>> covering({1: [], 2: [10], 3: [40]}, [], 45)
    0.8189300411522634

    """
    Ak = {
        k + 1: partition_from_cps(annotations[uid], n_obs)
        for k, uid in enumerate(annotations)
    }
    pX = partition_from_cps(predictions, n_obs)

    Cs = [cover_single(Ak[k], pX) for k in Ak]
    return sum(Cs) / len(Cs)


def sequence_data(data,K):

    """
    Splits data into sequence of equally long non-overlapping batches

    Parameters:
    - data (numpy array): Input data
    - K (int): Length of batches

    Returns:
    - list: List of data batches  
    """
    list_of_batches = []

    if not isinstance(data,np.ndarray):
        raise ValueError("data must be numpy array")
    
    if not isinstance(K,int):
        raise ValueError("K must be integer")
    
    T = data.shape[0]

    if T <= K: 
        raise ValueError("K must be smaller than length of time series")
    
    nr_batches_int = T // K

    for i in range(nr_batches_int):
        start_idx = i * K
        end_idx = (i+1) * K
        list_of_batches.append(data[start_idx:end_idx,:])
        
    return list_of_batches


def load_dataset(filename):
    """Load a CPDBecnh dataset"""
    with open(filename, "r") as fp:
        data = json.load(fp)

    if data["time"]["index"] != list(range(0,data["n_obs"])):
        raise NotImplementedError("Time series with non-consecutive time axis are not yet supported.")
    
    mat = np.zeros((data["n_obs"], data["n_dim"]))
    for j, series in enumerate(data["series"]):
        mat[:,j] = series["raw"]
        
    # We normalize to avoid numerical errors
    mat = (mat - np.nanmean(mat, axis=0)) / np.sqrt(np.nanvar(mat, axis=0, ddof=1))

    return data, mat



