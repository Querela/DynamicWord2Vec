# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 11:57:51 2017

@author: suny2
"""
import os

import numpy as np
import scipy.io as sio
import scipy.sparse.linalg as ssl


def get_word_idx(filename, skip_header=False):
    """Load input data and build lookup word to w_id and list of frequencies.

    :param filename: input filename for CSV file with w_id, word, frequency
    :param skip_header: whether to skip a possible header in filename (Default value = False)
    :returns: tuple: lookup word2w_id and frequencies for each word

    """
    vocab2id = dict()
    # id2freq = dict()
    freqlist = list()

    # assume ordered, w_id asc; and no duplicates
    with open(filename, "r", encoding="utf-8") as fid:
        if skip_header:
            fid.readline()
        for line in fid:
            line = line.strip("\n").split(",")

            w_id = int(line[0])
            word = line[1].strip()
            freq = int(line[2])

            vocab2id[word] = w_id
            # id2freq[w_id] = freq
            freqlist.append(freq)

    freqlist = np.array(freqlist)
    return vocab2id, freqlist


def get_cum_cooccur(filename, vocab2id, skip_header=True):
    """Build cummulative cooccurrence matrix.

    :param filename: input filename of yearly cooccurrences
    :param vocab2id: lookup word to w_id
    :param skip_header: whether to skip a possible header in filename (Default value = True)
    :returns: matrix with cooccurrence, sparse (but dense object)

    """
    num_words = len(vocab2id.keys())
    cooccur = np.zeros((num_words, num_words))

    with open(filename, "r", encoding="utf-8") as fid:
        if skip_header:
            fid.readline()

        for ln, line in enumerate(fid, 1):
            # if ln % 100000 == 0:
            #     print(ln / (41709765.0))

            counts = line.strip("\n").split(",")

            words = counts[0].split(":")
            w1_id = vocab2id[words[0]]
            w2_id = vocab2id[words[1]]

            if w1_id == w2_id:
                print("word-pair same?: '{}', '{}'".format(w1_id, w2_id))
                # continue

            for count in counts[1:]:
                if not count.strip():
                    continue
                cooccur[w1_id, w2_id] += int(count)
                cooccur[w2_id, w1_id] += int(count)

    return cooccur


def build_static_embs(cooccur, freq, rank=50):
    """Build eigen(values/vectors) embeddings and saves them.
    See: SMOP (not so good), converted from matlab. Hopefully works ...

    :param cooccur: cooccurence matrix
    :param freq: list of frequencies
    :param rank: rank/dimension for reduction? number of eigenvalues/eigenvectors

    """

    cooccur += np.diag(freq)  # cooccur = cooccur + diag(freq);
    cooccur *= np.sum(freq)  # cooccur = cooccur*sum(freq) ./ (freq*freq');
    cooccur /= np.dot(freq, freq.T)

    pmi = np.log(np.max(cooccur, 0))
    pmi[np.isinf(pmi)] = 0
    # asdf  # ??

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html#scipy.sparse.linalg.eigsh
    # https://de.mathworks.com/help/matlab/ref/eigs.html#bu2_q3e-sigma
    X, D = ssl.eigsh(
        pmi, rank, which="LA"
    )  # sigma='la'/largestreal option in Matlab? opts=isreal/issym

    DD = np.diag(D)
    DD = np.max(DD, 0)

    emb = np.dot(X, np.diag(np.sqrt(DD)))

    return X, D, emb


def main(data_dir="data"):
    """Main workflow.

    :param data_dir: directory with files

    """
    word_freq_file = "wordIDHash.csv"
    yearly_cooc_file = "wordCoOccurByYear_min200_ws5.csv"
    initial_coocfreq_file = "initial_cooccur_freq.mat"
    eigs_static_file = "eigs_static.mat"
    emb_static_file = "emb_static.mat"

    if os.path.exists(data_dir):
        word_freq_file = os.path.join(data_dir, word_freq_file)
        yearly_cooc_file = os.path.join(data_dir, yearly_cooc_file)
        initial_coocfreq_file = os.path.join(data_dir, initial_coocfreq_file)
        eigs_static_file = os.path.join(data_dir, eigs_static_file)
        emb_static_file = os.path.join(data_dir, emb_static_file)

    print("* Load words and frequencies ...")
    vocab2id, freqlist = get_word_idx(word_freq_file)
    print("* Load yearly coocs and build matrix ...")
    cooccur = get_cum_cooccur(yearly_cooc_file, vocab2id)

    sio.savemat(initial_coocfreq_file, {"cooccur": cooccur, "freq": freqlist})

    print("* Generate static embeddings ...")
    X, D, emb = build_static_embs(cooccur, freqlist, rank=50)

    sio.savemat(eigs_static_file, {"X": X, "D": D})  # save -v7.3 eigs_static X D
    sio.savemat(emb_static_file, {"emb": emb})


if __name__ == "__main__":
    main()
