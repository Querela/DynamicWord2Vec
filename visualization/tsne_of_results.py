# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09  2017


"""
import os.path
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# from scipy.spatial.distance import pdist
from sklearn.manifold import TSNE

# ---------------------------------------------------------------------------


def filter_embeddings_closest_words(
    word, emb_all, wordlist, word2idx, time_range, num_nearest_neighbors=50
):
    #: closest neighbor embeddings
    X = []
    #: list of closest neighbor words
    list_of_words = []
    #: bool for which is the first neighborhood word (self)?
    isword = []

    for year in time_range:
        emb = emb_all["U_%d" % time_range.index(year)]

        # normalize embeddings
        embnrm = np.reshape(np.sqrt(np.sum(emb ** 2, 1)), (emb.shape[0], 1))
        emb_normalized = np.divide(emb, np.tile(embnrm, (1, emb.shape[1])))
        # print('Embedding shape: {}'.format(emb_normalized.shape))

        # get closest neighbors
        vec = emb_normalized[word2idx[word], :]
        dists = np.dot(emb_normalized, vec)
        idx = np.argsort(dists)[::-1]

        newwords = [(wordlist[k], year) for k in list(idx[:num_nearest_neighbors])]
        print("Closest words in year {}: {}".format(year, [w[0] for w in newwords]))

        list_of_words.extend(newwords)
        for k in range(num_nearest_neighbors):
            isword.append(k == 0)
        X.append(emb[idx[:num_nearest_neighbors], :])

    X = np.vstack(X)
    print("Selected neighbor word embeddings shape: {}".format(X.shape))

    return X, list_of_words, isword


# ---------------------------------------------------------------------------


def project2D(X):
    model = TSNE(n_components=2, metric="euclidean")
    Z = model.fit_transform(X)
    return Z


# ---------------------------------------------------------------------------


def plot_all(
    Z,
    list_of_words,
    isword,
    figure_file,
    show_neighbors=True,
    show_only_isword_labels=False,
):
    plt.clf()
    plt.axis("off")

    traj = []
    for k, word in enumerate(list_of_words):
        if isword[k]:
            marker = "ro"
            traj.append(Z[k, :])
        elif show_neighbors:
            marker = "b."

        plt.plot(Z[k, 0], Z[k, 1], marker)
        if not show_only_isword_labels or isword[k]:
            txt = plt.text(Z[k, 0], Z[k, 1], " %s-%d" % (word[0], word[1]))
            if not isword[k]:
                txt.set_alpha(0.4)

    traj = np.vstack(traj)
    plt.plot(traj[:, 0], traj[:, 1])

    plt.savefig(figure_file)


# ---------------------------------------------------------------------------


def plot_trajectory(Z, list_of_words, isword, figure_file):
    # rescale y-axis?
    Zp = Z * 1.0
    Zp[:, 0] = Zp[:, 0] * 2.0

    # nxn dists
    num_points = Z.shape[0]
    all_dist = np.zeros((num_points, num_points))
    for k in range(num_points):
        all_dist[:, k] = np.sum(
            (Zp - np.tile(Zp[k, :], (num_points, 1))) ** 2.0, axis=1
        )

    dist_to_centerpoints = all_dist[:, isword]
    dist_to_centerpoints = np.min(dist_to_centerpoints, axis=1)

    dist_to_other = all_dist + np.eye(num_points) * 1000.0
    idx_dist_to_other = np.argsort(dist_to_other, axis=1)
    dist_to_other = np.sort(dist_to_other, axis=1)  # TODO: resort with indices?

    # ------------------------------------

    plt.clf()
    plt.axis("off")

    traj = []
    for k in reversed(range(len(list_of_words))):
        if isword[k]:
            # if list_of_words[k][1] % 30 != 0 and list_of_words[k][1] < 1990: continue
            marker = "bo"
            traj.append(Z[k, :])
            plt.plot(Z[k, 0], Z[k, 1], marker)
        else:
            # too far from center
            if dist_to_centerpoints[k] > 200:
                continue

            skip = False
            for i in range(Z.shape[0]):
                # ?
                if dist_to_other[k, i] < 150 and idx_dist_to_other[k, i] > k:
                    skip = True
                    break
                if dist_to_other[k, i] >= 150:
                    break

            if skip:
                continue

            if Z[k, 0] > 8:
                continue

            plt.plot(Z[k, 0], Z[k, 1])

        plt.text(
            Z[k, 0] - 2,
            Z[k, 1] + np.random.randn() * 2,
            " %s-%d" % (list_of_words[k][0], list_of_words[k][1]),
        )

    traj = np.vstack(traj)
    plt.plot(traj[:, 0], traj[:, 1])

    plt.savefig(figure_file)


# ---------------------------------------------------------------------------


def main(
    word, times, wordlist_filename, embeddings_filename, output_dir, recompute=False
):
    print('Run for word "{}" in time {}'.format(word, times))
    sane_word = "".join([c for c in word if re.match(r"\w", c)])
    state_emb_file = os.path.join(output_dir, "{}_tsne.mat".format(sane_word))
    state_wrd_file = os.path.join(output_dir, "{}_tsne_wordlist.pkl".format(sane_word))
    figure_tsne_file = os.path.join(output_dir, "{}_traj_much.png".format(sane_word))
    figure_traj_file = os.path.join(output_dir, "{}_traj.png".format(sane_word))

    wordlist = []
    with open(wordlist_filename, "r", encoding="utf-8") as fid:
        wordlist = [word_.strip() for word_ in fid]
    # num_words = len(wordlist)

    word2idx = {word_: w_id for w_id, word_ in enumerate(wordlist)}

    # if not previously saved state exists, then build and save it
    if (
        not recompute
        and os.path.exists(state_emb_file)
        and os.path.exists(state_wrd_file)
    ):
        print("* Load pre-computed data ...")
        #: re-load data
        Z = sio.loadmat(state_emb_file)["emb"]
        with open(state_wrd_file, "rb") as fid:
            data = pickle.load(fid)
        list_of_words, isword = data["words"], data["isword"]
    else:
        print("* Build 2D embedding projection ...")
        emb_all = sio.loadmat(embeddings_filename)
        X, list_of_words, isword = filter_embeddings_closest_words(
            word, emb_all, wordlist, word2idx, times, 50
        )
        Z = project2D(X)

        #: persist data
        sio.savemat(state_emb_file, {"emb": Z})
        with open(state_wrd_file, "wb") as fid:
            pickle.dump({"words": list_of_words, "isword": isword}, fid)

    print("* Plot embedding projection ...")
    plot_all(
        Z,
        list_of_words,
        isword,
        figure_tsne_file,
        show_neighbors=True,
        show_only_isword_labels=True,
    )

    print("* Plot trajectory ...")
    plot_trajectory(Z, list_of_words, isword, figure_traj_file)


if __name__ == "__main__":
    word = "communist"
    times = range(
        1800, 2000, 10
    )  # total number of time points (20/range(27) for ngram/nyt)

    # TODO: unused
    allwords = ["art", "damn", "gay", "hell", "maid", "muslim"]

    wordlist_filename = "data/wordlist.txt"
    embeddings_filename = "results/embeddings.mat"
    output_dir = "tsne_output"

    main(
        word, times, wordlist_filename, embeddings_filename, output_dir, recompute=False
    )

    for word_ in allwords:
        main(
            word_,
            times,
            wordlist_filename,
            embeddings_filename,
            output_dir,
            recompute=False,
        )

    print("Done.")
