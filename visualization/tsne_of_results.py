# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09  2017


"""
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from pprint import pprint
from scipy.spatial.distance import pdist
from sklearn.manifold import TSNE

# ---------------------------------------------------------------------------

word = "communist"

wordlist = []
with open("data/wordlist.txt", "r") as fid:
    for line in fid:
        wordlist.append(line.strip())
nw = len(wordlist)

word2Id = {}
for w_id, word in enumerate(wordlist):
    word2Id[word] = w_id

times = range(180, 200)  # total number of time points (20/range(27) for ngram/nyt)

emb_all = sio.loadmat("results/embeddings.mat")

# ---------------------------------------------------------------------------

nn = 50  # number of nearest neighbors

emb = emb_all["U_%d" % times.index(199)]

X = []
list_of_words = []
isword = []

for year in times:
    emb = emb_all["U_%d" % times.index(year)]
    embnrm = np.reshape(np.sqrt(np.sum(emb ** 2, 1)), (emb.shape[0], 1))
    emb_normalized = np.divide(emb, np.tile(embnrm, (1, emb.shape[1])))
    print(emb_normalized.shape)

    v = emb_normalized[word2Id[word], :]
    d = np.dot(emb_normalized, v)
    idx = np.argsort(d)[::-1]
    newwords = [(wordlist[k], year) for k in list(idx[:nn])]
    print(newwords)
    list_of_words.extend(newwords)

    for k in range(nn):
        isword.append(k == 0)
    X.append(emb[idx[:nn], :])
    # print(year, [wordlist[i] for i in idx[:nn]])

X = np.vstack(X)
print(X.shape)

# ---------------------------------------------------------------------------

model = TSNE(n_components=2, metric="euclidean")
Z = model.fit_transform(X)

# ---------------------------------------------------------------------------

plt.clf()

traj = []
for k, word in enumerate(list_of_words):
    if isword[k]:
        marker = "ro"
        traj.append(Z[k, :])
    else:
        marker = "b."

    plt.plot(Z[k, 0], Z[k, 1], marker)
    plt.text(Z[k, 0], Z[k, 1], word)

traj = np.vstack(traj)
plt.plot(traj[:, 0], traj[:, 1])
plt.show()
plt.savefig('tsne_output/traj_much.png')

sio.savemat("tsne_output/%s_tsne.mat" % word, {"emb": Z})
pickle.dump(
    {"words": list_of_words, "isword": isword},
    open("tsne_output/%s_tsne_wordlist.pkl" % word, "wb"),
)

# ---------------------------------------------------------------------------

allwords = ["art", "damn", "gay", "hell", "maid", "muslim"]

Z = sio.loadmat("tsne_output/%s_tsne.mat" % word)["emb"]
data = pickle.load(open("tsne_output/%s_tsne_wordlist.pkl" % word, "rb"))
list_of_words, isword = data["words"], data["isword"]

Zp = Z * 1.0
Zp[:, 0] = Zp[:, 0] * 2.0
all_dist = np.zeros((Z.shape[0], Z.shape[0]))
for k in range(Z.shape[0]):
    all_dist[:, k] = np.sum((Zp - np.tile(Zp[k, :], (Z.shape[0], 1))) ** 2.0, axis=1)

dist_to_centerpoints = all_dist[:, isword]
dist_to_centerpoints = np.min(dist_to_centerpoints, axis=1)

dist_to_other = all_dist + np.eye(Z.shape[0]) * 1000.0
idx_dist_to_other = np.argsort(dist_to_other, axis=1)
dist_to_other = np.sort(dist_to_other, axis=1)

plt.clf()

traj = []
for k in range(len(list_of_words) - 1, -1, -1):
    if isword[k]:
        # if list_of_words[k][1] % 3 != 0 and list_of_words[k][1] < 199 : continue
        marker = "bo"
        traj.append(Z[k, :])
        plt.plot(Z[k, 0], Z[k, 1], marker)
    else:
        if dist_to_centerpoints[k] > 200:
            continue
        skip = False
        for i in range(Z.shape[0]):
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
        " %s-%d" % (list_of_words[k][0], list_of_words[k][1] * 10),
    )

plt.axis("off")
traj = np.vstack(traj)
plt.plot(traj[:, 0], traj[:, 1])
plt.show()
plt.savefig('tsne_output/traj.png')
