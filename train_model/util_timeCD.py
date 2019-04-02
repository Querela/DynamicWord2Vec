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
# in paper: eq(8), V == W

import copy

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as ss
from sklearn.metrics.pairwise import cosine_similarity


def update(U, Y, Vm1, Vp1, lam, tau, gam, ind, iflag):
    """equations are to update V

    :param U: embeddings for U to update V (?), at timestep (t) [???]
    :param Y: PMI Word-Matrix
    :param Vm1: embeddings V (?), at timestep (t - 1) (previous)
    :param Vp1: embeddings V (?), at timestep (t + 1) (next)
    :param lam: frob regularizer
    :param tau: smoothing regularizer / time regularizer
    :param gam: forcing regularizer / symmetry regularizer
    :param ind: batching indices
    :param iflag: indicator for first/last timestep, adjusted constants

    """
    #: U is n X r
    UtU = np.dot(U.T, U)  # rxr
    #: r = rank
    rank = UtU.shape[0]  # r

    if iflag:
        M = UtU + (lam + 2 * tau + gam) * np.eye(rank)
    else:
        M = UtU + (lam + tau + gam) * np.eye(rank)

    #: Y is n X b (b = batch size)
    Uty = np.dot(U.T, Y)  # rxb
    Ub = U[ind, :].T  # rxb

    #: Vm1 and Vp1 are bXr, so they are b rows of V, transposed
    A = Uty + gam * Ub + tau * (Vm1.T + Vp1.T)  # rxb
    Vhat = np.linalg.lstsq(M, A, rcond=None)  # rxb  # rcond=None to silence warning
    return Vhat[0].T  # bxr


def import_static_init(data_file, times):
    """Load MATLAB file with embeddings in "emb" to initialize embeddings with a given state?

    :param data_file: MATLAB file
    :param times: number of timepoints
    :returns: embeddings U, V for each timepoint (same)

    """
    # (T = times)
    emb = sio.loadmat(data_file)["emb"]
    U = [copy.deepcopy(emb) for t in times]
    V = [copy.deepcopy(emb) for t in times]
    return U, V


def initvars(vocab_size, times, rank):
    """Initialize embeddings randomly, for each time point the same.

    :param vocab_size: number of words in vocabulary
    :param times: number of timepoints as range object
    :type times: range()
    :param rank: embedding dimension (?)
    :returns: embeddings matrices U, V randomly initialized

    """
    # (T = times)
    # dictionary will store the variables U and V.
    #   tuple (t, i) indexes time t and word index i
    U, V = [], []
    U.append(np.random.randn(vocab_size, rank) / np.sqrt(rank))
    V.append(np.random.randn(vocab_size, rank) / np.sqrt(rank))
    for t in times:
        U.append(U[0].copy())
        V.append(V[0].copy())
        # print(t)
    return U, V


def save_embeddings(data_file, emb):
    """Save embeddings in MATLAB data file.

    :param data_file: filepath to store the embeddings in MATLAB file at.
    :param emb: embeddings to store

    """
    sio.savemat(data_file, {"emb": emb})


def save_embeddings_split(data_file, emb, time_range, prefix="U_"):
    """Save embeddings in MATLAB data file. Split per timepoint.
    Add keys as combination of `prefix` and index in `time_range`.

    :param data_file: filepath to store the embeddings in MATLAB file at.
    :param emb: embeddings to store
    :param time_range: range of timepoints
    :param prefix: key prefix (Default value = "U_")

    """
    if not prefix:
        prefix = ""

    embs = dict()
    for tx, time in enumerate(time_range):
        key = "{}{}".format(prefix, tx)
        embs[key] = emb[tx]

    sio.savemat(data_file, embs)


def getmat(filename, vocab_size, rowflag=False, make_dense=False, inds=None):
    """Load PMI (?) matrix from a given filepath (CSV file)

    :param filename: CSV file with <w1_id><w2_id><pmi_value>
    :param vocab_size: size of vocabulary, for nxn sparse matrix
    :param rowflag: sparse matrix in row or column order (?), (Default value = False)
    :param make_dense: keep sparse matrix or make dense (Default value = False)
    :param inds: batching indices, return subset of data if not None (Default value = None)

    """
    data = pd.read_csv(filename)
    data = data.values

    X = ss.coo_matrix(
        (data[:, 2], (data[:, 0], data[:, 1])), shape=(vocab_size, vocab_size)
    )

    if rowflag:
        X = ss.csr_matrix(X)
        if inds is not None:
            X = X[inds, :]
    else:
        X = ss.csc_matrix(X)
        if inds is not None:
            X = X[:, inds]

    if make_dense:
        X = X.todense()

    return X


def getbatches(vocab_size, batch_size):
    """Make batching indices to be able to iteratively work with large matrices.

    :param vocab_size: number of words in vocabulary
    :param batch_size: size per batch, should be smaller or equal to vocab_size
    :returns: list with ranges of indices

    """
    batch_inds = []
    current = 0
    while current < vocab_size:
        end = min(current + batch_size, vocab_size)
        batch_inds.append(range(current, end))
        current = end
    return batch_inds


def getclosest(wid, U, num_results=10):
    """Takes a word id and returns closest words by cosine distance.

    :param wid: word id
    :param U: embedding matrix
    :param num_results: number of closest words to return (Default value = 10)
    :returns: list for each timepoint with list of num_results closest neighbor words

    """
    C = []
    # for each timestep
    for t in range(len(U)):
        temp = U[t]
        # K = cosine_similarity(temp[wid, :], temp)
        K = np.dot(temp, temp[wid, :])
        mxinds = np.argsort(-K)
        mxinds = mxinds[0:num_results]
        C.append(mxinds)
    return C


def compute_symscore(U, V):
    """Computes the regularizer scores given U and V entries

    :param U: embeddings
    :param V: embeddings
    :returns: symscore

    """
    return np.linalg.norm(U - V) ** 2


def compute_smoothscore(U, Um1, Up1):
    """? Assume, to be the difference between previous and next timepoints.

    :param U: embedding matix at timepoint t
    :param Up1: embedding matrix at timepoint (t - 1)
    :param Um1: embedding matrix at timepoint (t + 1)
    :returns: smoothscore

    """
    return np.linalg.norm(U - Up1) ** 2 + np.linalg.norm(U - Um1) ** 2
