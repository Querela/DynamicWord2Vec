# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 11:57:51 2017

@author: suny2
"""
import datetime
import os
import time

import numpy as np
import scipy.io as sio
import scipy.sparse.linalg as ssl


def _get_time_diff(start_time, end_time=None):
    """Makes a `datetime.timedelta` object from a time difference.

    :param start_time: start time value (from `time.time()`)
    :param end_time: optional end time, or will retrieve current time (Default value = None)
    :returns: time delta
    :rtype: datetime.timedelta`

    """
    if end_time is None:
        end_time = time.time()

    time_diff = end_time - start_time
    time_diff = int(time_diff)
    time_delta = datetime.timedelta(seconds=time_diff)

    return time_delta


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
            line = line.strip("\n").split("\t")  # ","

            w_id = int(line[0])
            word = line[1].strip()
            freq = int(line[2])

            # this can silently ignore duplicate words ... so:
            if word in vocab2id:
                print(
                    "  ! word duplicate: {}, ids: {},{}".format(
                        word, vocab2id[word], w_id
                    )
                )
                continue

            vocab2id[word] = w_id

            # id2freq[w_id] = freq
            freqlist.append(freq)

    freqlist = np.array(freqlist)

    # if it fails, you have to modify manually ...
    # assert freqlist.shape[0] != len(vocab2id), "duplicate words?"

    return vocab2id, freqlist


def get_cum_cooccur(filename, vocab2id, skip_header=True):
    """Build cummulative cooccurrence matrix from CSV data file.

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

        num_err = 0
        for ln, line in enumerate(fid, 1):
            # if ln % 100000 == 0:
            #     print(ln / (41709765.0))

            word1, word2, counts = line.strip("\n").split("\t")

            try:
                w1_id = vocab2id[word1]
                w2_id = vocab2id[word2]
            except KeyError as ex:
                num_err += 1
                if num_err < 10:
                    print("Line: {} - {}".format(ln, ex))

            if w1_id == w2_id:
                print("word-pair same?: '{}', '{}'".format(w1_id, w2_id))
                # continue

            for count in counts.split(","):
                if not count.strip():
                    continue
                cooccur[w1_id, w2_id] += int(count)
                # cooccur[w2_id, w1_id] += int(count)

        print("{} errors.".format(num_err))

    return cooccur


def build_static_embs(cooccur, freq, rank=50, debug=False):
    """Build eigen(values/vectors) embeddings.
    See: SMOP (not so good), converted from matlab. Hopefully works ...

    :param cooccur: cooccurence matrix
    :param freq: vector of frequencies
    :param rank: rank/dimension for reduction? number of eigenvalues/eigenvectors (Default value = 50)
    :param debug: print some debug information (Default value = False)
    :returns: tuple: eigenvectors, eigenvalues, embeddings

    """
    if debug:
        print("cooccur shape = {}".format(cooccur.shape))
        print("freq shape = {}".format(freq.shape))

    cooccur += np.diag(freq)  # cooccur = cooccur + diag(freq);
    if debug:
        print("cooccur: {}".format(cooccur.shape))
    cooccur *= np.sum(freq)  # cooccur = cooccur*sum(freq) ./ (freq*freq');
    if debug:
        print("cooccur: {}".format(cooccur.shape))
    cooccur /= np.dot(freq, freq.T)
    if debug:
        print("cooccur: {}".format(cooccur.shape))

    pmi = np.log(np.clip(cooccur, 0, None))
    pmi[np.isinf(pmi)] = 0
    if debug:
        print("pmi shape = {}".format(pmi.shape))
    # asdf  # ??

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html#scipy.sparse.linalg.eigsh
    # https://de.mathworks.com/help/matlab/ref/eigs.html#bu2_q3e-sigma
    # eigs(...) (square matrix) - eigsh(...) (sq.mat, real symmetric)
    eigenvalues, eigenvectors = ssl.eigs(
        pmi, rank, which="LA"
    )  # sigma='la'/largestreal option in Matlab? opts=isreal/issym
    # MATLAB: [X, D] = eigs(pmi,50,'la', opts);
    # X - eigenvectors, columns are the eigenvectors of A such that A*X = X*D
    # D - eigenvalues, diagonal matrix with the eigenvalues on the main diagonal
    # SCIPY: (eigenvalues, eigenvectors)
    # eigenvalues - eigenvalues, array
    # eigenvectors - eigenvectors, column eigenvectors[:, i] is the eigenvector corresponding to the eigenvalue eigenvalues[i]

    if debug:
        print("eigenvalues shape = {}".format(eigenvalues.shape))
        print("eigenvectors shape = {}".format(eigenvectors.shape))

    # sqrt, above zero
    eigenvalues2 = np.sqrt(np.clip(eigenvalues, 0, None))
    if debug:
        print("eigenvalues2: {}".format(eigenvalues2.shape))

    emb = np.dot(eigenvectors, np.diag(eigenvalues2))
    if debug:
        print("emb shape = {}".format(emb.shape))

    return eigenvectors, eigenvalues, emb


def main(
    word_freq_file,
    yearly_cooc_file,
    emb_static_file,
    rank,
    initial_coocfreq_file=None,
    eigs_static_file=None,
    debug=False,
):
    """Main workflow.

    :param word_freq_file: csv file with w_id,word,frequency
    :param yearly_cooc_file: csv file with word-pair and yearly value (pmi?)
    :param emb_static_file: output file with initial static embeddings
    :param rank: rank/dimension of embeddings (number of eigenvectors to compute)
    :param initial_coocfreq_file: output file for cooc matrix and frequency vector (Default value = None)
    :param eigs_static_file: output file for eigenvector/eigenvalue data (Default value = None)
    :param debug: debug output information (Default value = False)

    """
    print("* Load words and frequencies ...", end="", flush=True)
    start_time = time.time()
    vocab2id, freqlist = get_word_idx(word_freq_file)
    print(" {}".format(_get_time_diff(start_time)))

    print("* Load yearly coocs and build matrix ...", end="", flush=True)
    start_time = time.time()
    cooccur = get_cum_cooccur(yearly_cooc_file, vocab2id)
    print(" {}".format(_get_time_diff(start_time)))

    if initial_coocfreq_file:
        print("  * Save cooc mat + freq vec to: {}".format(initial_coocfreq_file))
        try:
            sio.savemat(initial_coocfreq_file, {"cooccur": cooccur, "freq": freqlist})
        except Exception as ex:
            print("    ! {}".format(ex))

    print("* Generate static embeddings ...")
    start_time = time.time()
    eigenvectors, eigenvalues, emb = build_static_embs(
        cooccur, freqlist, rank=rank, debug=debug
    )
    print("~ Took {}".format(_get_time_diff(start_time)))

    if eigs_static_file:
        print("  * Save eigenvectors/-values to: {}".format(eigs_static_file))
        sio.savemat(
            eigs_static_file, {"X": eigenvectors, "D": eigenvalues}
        )  # save -v7.3 eigs_static X D
    print("  * Save static embeddings to: {}".format(emb_static_file))
    sio.savemat(emb_static_file, {"emb": emb})


def parse_args():
    """Parse arguments, use defaults if not set.

    :returns: arguments/parameters

    """
    # Defaults

    #: directory
    data_dir = "data"

    #: inputs
    words_file = "wordIDHash.csv"
    yearly_coocs_file = "wordCoOccurByYear_min200_ws5.csv"

    #: results/outputs
    coocs_matrix_file = "initial_cooccur_freq.mat"
    eigs_static_file = "eigs_static.mat"
    emb_static_file = "emb_static.mat"

    rank = 50

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--data-dir",
        default=data_dir,
        help="base directory for all input and output files, default: {}".format(
            data_dir
        ),
    )
    parser.add_argument(
        "-w",
        "--words-file",
        default=words_file,
        help="input filename for CSV file of words, w_ids and frequencies, default: {}".format(
            words_file
        ),
    )
    parser.add_argument(
        "-c",
        "--yearly-coocs-file",
        default=yearly_coocs_file,
        help="input filename for CSV file of yearly cooccurrence data, default: {}".format(
            yearly_coocs_file
        ),
    )
    parser.add_argument(
        "-e",
        "--emb-file",
        default=emb_static_file,
        help="output filename for initial static embeddings, default: {}".format(
            emb_static_file
        ),
    )
    parser.add_argument(
        "-r",
        "--rank",
        type=int,
        default=rank,
        help="rank/dimensions for eigenvector computation and resulting embeddings, default: {}".format(
            rank
        ),
    )
    parser.add_argument(
        "--coocs-matrix-file",
        default=coocs_matrix_file,
        help="output filename for matrix with word cooccurence frequencies?; empty to disable, default: {}".format(
            coocs_matrix_file
        ),
    )
    parser.add_argument(
        "--eigs-file",
        default=eigs_static_file,
        help="output filename for eigenvalue/eigenvector data; empty to disable, default: {}".format(
            eigs_static_file
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="debugging information about matix shapes etc.",
    )

    args = parser.parse_args()

    if not args.coocs_matrix_file:
        args.coocs_matrix_file = None
    if not args.eigs_file:
        arsg.eigs_file = None

    if os.path.exists(args.data_dir):
        args.words_file = os.path.join(args.data_dir, args.words_file)
        args.yearly_coocs_file = os.path.join(args.data_dir, args.yearly_coocs_file)

        if args.coocs_matrix_file:
            args.coocs_matrix_file = os.path.join(args.data_dir, args.coocs_matrix_file)
        if args.eigs_file:
            args.eigs_file = os.path.join(args.data_dir, args.eigs_file)
        args.emb_file = os.path.join(args.data_dir, args.emb_file)

    return args


if __name__ == "__main__":
    args = parse_args()
    main(
        args.words_file,
        args.yearly_coocs_file,
        args.emb_file,
        args.rank,
        args.coocs_matrix_file,
        args.eigs_file,
        debug=args.debug,
    )
