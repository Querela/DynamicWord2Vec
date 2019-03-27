#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 10:11:23 2016

@author: raon
"""

# utility functions for running the CD method
# loss: min 1/2 \sum_t | Yt - UtVt' |^2 + lam/2 \sum_t    (|Ut|^2        + |Vt|^2)
#                                       + tau/2 \sum_t >1 (|Vt - Vt-1|^2 + |Ut - Ut-1|^2)
#                                       + gam/2 \sum_t    (|Ut - Vt|^2)

import copy

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as ss
from sklearn.metrics.pairwise import cosine_similarity


def update(U, Y, Vm1, Vp1, lam, tau, gam, ind, iflag):
    UtU = np.dot(U.T, U)  # rxr
    rank = UtU.shape[0]  # r

    if iflag:
        M = UtU + (lam + 2 * tau + gam) * np.eye(rank)
    else:
        M = UtU + (lam + tau + gam) * np.eye(rank)

    Uty = np.dot(U.T, Y)  # rxb
    Ub = U[ind, :].T  # rxb

    A = Uty + gam * Ub + tau * (Vm1.T + Vp1.T)  # rxb
    Vhat = np.linalg.lstsq(M, A)  # rxb
    return Vhat[0].T  # bxr


# for the above function, the equations are to update V. So:
# Y is n X b (b = batch size)
# r = rank
# U is n X r
# Vm1 and Vp1 are bXr. so they are b rows of V, transposed


def import_static_init(times):
    # (T = times)
    emb = sio.loadmat("data/emb_static.mat")["emb"]
    U = [copy.deepcopy(emb) for t in times]
    V = [copy.deepcopy(emb) for t in times]
    return U, V


def initvars(vocab_size, times, rank):
    # (T = times)
    # dictionary will store the variables U and V.
    #   tuple (t, i) indexes time t and word index i
    U, V = [], []
    U.append(np.random.randn(vocab_size, rank) / np.sqrt(rank))
    V.append(np.random.randn(vocab_size, rank) / np.sqrt(rank))
    for t in times:
        U.append(U[0].copy())
        V.append(V[0].copy())
        print(t)
    return U, V


def getmat(filename, vocab_size, rowflag):
    data = pd.read_csv(filename)
    data = data.values

    X = ss.coo_matrix((data[:, 2], (data[:, 0], data[:, 1])),
                      shape=(vocab_size, vocab_size))

    if rowflag:
        X = ss.csr_matrix(X)
        # X = X[inds,:]
    else:
        X = ss.csc_matrix(X)
        # X = X[:,inds]

    return X  # .todense()


def getbatches(vocab_size, batch_size):
    batch_inds = []
    current = 0
    while current < vocab_size:
        end = min(current + batch_size, vocab_size)
        batch_inds.append(range(current, end))
        current = end
    return batch_inds

 
def getclosest(wid, U, num_results=10):
    '''takes a word id and returns closest words by cosine distance'''
    C = []
    # for each timestep
    for t in range(len(U)):
        temp = U[t]
        K = cosine_similarity(temp[wid, :], temp)
        mxinds = np.argsort(-K)
        mxinds = mxinds[0:num_results]
        C.append(mxinds)
    return C


def compute_symscore(U, V):
    '''computes the regularizer scores given U and V entries'''
    return np.linalg.norm(U - V) ** 2


def compute_smoothscore(U, Um1, Up1):
    return np.linalg.norm(U - Up1) ** 2 + np.linalg.norm(U - Um1) ** 2
