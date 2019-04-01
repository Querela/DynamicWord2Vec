#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script to generate subsampled datasets etc

Created on Tue Jan 17 20:23:29 2017

@author: raon
"""

import os

import numpy as np
import pandas as pd
import numpy.random as nr
import scipy.sparse as ss


"""
DynamicWord2Vec/data/NYTimesV2/wordPairPMI_<0-27>.csv
"""


#: TODO
def yearmap(time_range, startyear=1990):
    # list containing year numbers
    endyear = startyear + len(time_range)
    return list(range(startyear, endyear))


def read_data(filename, num_words):
    data = pd.read_csv(filename)
    data = data.values

    X = ss.coo_matrix(
        (data[:, 2], (data[:, 0], data[:, 1])), shape=(num_words, num_words)
    )
    return X


def read_vocab(vfile):
    # list containing vocabulary
    vocab = []
    with open(vfile) as fid:
        for line in fid:
            vals = line.strip("\n").split(",")
            vocab.append(vals[1])
    return vocab


def subsample(X, percentage):
    data = X.data
    rows = X.row
    cols = X.col

    N = len(cols)
    shuf = list(nr.permutation(range(N)))
    cut = int(np.floor(N * percentage))

    rows = rows[shuf[0:cut]]
    cols = cols[shuf[0:cut]]
    data = data[shuf[0:cut]]
    Y = ss.coo_matrix((data, (rows, cols)), shape=X.shape)

    return Y


def remove_word(X, widx):
    X = ss.csr_matrix(X)
    X[widx, :] = 0

    X = ss.csc_matrix(X)
    X[:, widx] = 0

    X = ss.coo_matrix(X)
    return X


def get_id(vocab, word):
    widx = vocab.index(word)
    return widx


def random_subsample_all(fhead, percentage, time_range, num_words):
    for tx, time_point in enumerate(time_range):
        print(
            "Subsample {} [{}] ({}/{}) ...".format(
                time_point, tx, tx + 1, len(time_range)
            ),
            end="",
            flush=True,
        )
        filename = "{}{}.csv".format(fhead, tx)
        X = read_data(filename, num_words)

        Y = subsample(X, percentage)
        data = Y.data
        rows = Y.row
        cols = Y.col
        M = np.hstack((data, rows, cols))

        filename = "{}{}_S_{}.csv".format(fhead, tx, percentage)
        np.savetxt(filename, M, delimiter=",")
        print(" done.")


def filter_word_all(fhead, word, years, time_range, vocab, num_words):
    allyears = yearmap(time_range)
    wid = get_id(vocab, word)

    for yn, year in enumerate(years, 1):
        print(
            "Filter year {} ({}/{}) ...".format(year, yn, len(years)),
            end="",
            flush=True,
        )
        time_point = time_range[allyears.index(year)]
        filename = "{}{}.csv".format(fhead, time_point)
        X = read_data(filename, num_words)

        Y = remove_word(X, wid)

        filename = "{}{}_R_{}.csv".format(fhead, time_point, word)
        np.savetxt(filename, Y, delimiter=",")
        print(" done.")


def main():
    base_dir = "DynamicWord2Vec/data/NYTimesV2"

    fhead = "wordPairPMI_"
    fhead = os.path.join(base_dir, fhead)

    num_words = 20936
    vocabfile = "wordIDHash.csv"
    vocabfile = os.path.join(base_dir, vocabfile)

    time_range = range(1990, 2017)  # 1990 -- 2016, len=27

    vocab = read_vocab(vocabfile)

    percentage = 0.2
    random_subsample_all(fhead, percentage, time_range, num_words)
    print("Subsampling {} done".format(percentage))

    percentage = 0.8
    random_subsample_all(fhead, percentage, time_range, num_words)
    print("Subsampling {} done".format(percentage))

    word = "apple"
    years = range(2010, 2013)
    filter_word_all(fhead, word, years, time_range, vocab, num_words)


if __name__ == "__main__":
    main()
