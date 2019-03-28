# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 11:57:51 2017

@author: suny2
"""
import numpy as np
import scipy.io as sio


def get_word_idx():
    vocab2id = {}
    id2freq = {}
    fid = open("wordIDHash.csv", "r")
    for line in fid:
        line = line.strip("\n").split(",")
        lid = int(line[0])
        freq = int(line[2])

        vocab2id[line[1].strip()] = lid
        id2freq[lid] = freq

    fid.close()
    freqlist = []
    for k in range(len(id2freq)):
        freqlist.append(id2freq[k])
    freqlist = np.array(freqlist)
    return vocab2id, freqlist


def get_cum_cooccur(vocab2id):
    nw = len(freqlist)
    cooccur = np.zeros((nw, nw))
    fid = open("wordCoOccurByYear_min200_ws5.csv", "r")
    header = fid.readline()
    linecount = 0
    for line in fid:
        linecount += 1
        if linecount % 100000 == 0:
            print(linecount / (41709765.0))
        counts = line.strip("\n").split(",")
        words = counts[0].split(":")
        i = vocab2id[words[0]]
        j = vocab2id[words[1]]
        if i == j:
            print(i, j)
            asdf
        counts = counts[1:]
        for k in range(len(counts)):
            if len(counts[k].strip()) == 0:
                continue
            cooccur[i, j] += int(counts[k])
            cooccur[j, i] += int(counts[k])
    fid.close()
    return cooccur


vocab2id, freqlist = get_word_idx()
cooccur = get_cum_cooccur(vocab2id)

sio.savemat("initial_cooccur_freq.mat", {"cooccur": cooccur, "freq": freqlist})
