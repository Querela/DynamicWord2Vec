# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 10:56:59 2017
Created on Mon Jan 09 20:46:52 2017

@author: suny2
"""
import itertools
import re

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


# ---------------------------------------------------------------------------


def compute_norms(words, emb_all, words2idx, time_range):
    allnorms = []
    for word in words:
        w_id = words2idx[word]
        norms = []
        for year in time_range:
            emb = emb_all["U_%d" % time_range.index(year)]
            vec = emb[w_id, :]
            norms.append(np.linalg.norm(vec))

        norms = np.array(norms)
        norms /= sum(norms)
        allnorms.append(norms)

    return allnorms


# ---------------------------------------------------------------------------


def plot_norms(allnorms, words, time_range, plot_filename):
    plt.clf()

    markers = itertools.cycle(["+", "o", "x", "*"])
    for k, (norms, marker) in enumerate(zip(allnorms, markers)):
        plt.plot(time_range, norms, marker=marker, markersize=7)

    plt.legend(words)
    plt.xlabel("year")
    plt.ylabel("word norm")

    plt.savefig(plot_filename)


# ---------------------------------------------------------------------------


def main(words, time_range, wordlist_filename, embeddings_filename, plot_filename):
    with open(wordlist_filename, "r", encoding="utf-8") as fid:
        wordlist = [word.strip() for word in fid]
    words2idx = {word: w_id for w_id, word in enumerate(wordlist)}

    emb_all = sio.loadmat(embeddings_filename)

    # ------------------------------------

    allnorms = compute_norms(words, emb_all, words2idx, time_range)

    plot_norms(allnorms, words, time_range, plot_filename)


if __name__ == "__main__":
    time_range = range(
        1800, 2000, 10
    )  # total number of time points (20/range(27) for ngram/nyt)

    words = ["thou", "chaise", "darwin", "telephone"]

    sane_words = ["".join([c for c in word if re.match(r"\w", c)]) for word in words]

    wordlist_filename = "data/wordlist.txt"
    # "results/emb_frobreg10_diffreg50_symmreg10_iter10.mat"
    embeddings_filename = "results/embeddings.mat"
    plot_filename = "{}_norm_plots.png".format("-".join(sane_words))

    main(words, time_range, wordlist_filename, embeddings_filename, plot_filename)
