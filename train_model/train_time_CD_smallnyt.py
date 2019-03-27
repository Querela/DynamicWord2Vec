#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:10:42 2016

"""

# main script for time CD
# trainfile has lines of the form
# tok1,tok2,pmi

import sys
import time

import numpy as np
import util_timeCD as util
import pickle as pickle

# PARAMETERS

num_words = 20936  # number of words in vocab (11068100/20936 for ngram/nyt)
T = range(1990, 2016)  # total number of time points (20/range(27) for ngram/nyt)

trainhead = "data/wordPairPMI_"  # location of training data
savehead = "results/"

SAVEPOINT_ITERATION = True
SAVEPOINT_ITER_TIME = False


def print_params(rank, lam, tau, gam, emph, ITERS):
    print("rank = {}".format(rank))
    print("frob regularizer = {}".format(lam))
    print("time regularizer = {}".format(tau))
    print("symmetry regularizer = {}".format(gam))
    print("emphasize param = {}".format(emph))
    print("total iterations = {}".format(ITERS))


def try_load_UV(savefile, iteration):
    Ulist = Vlist = None

    try:
        with open("{}ngU_iter{}.p".format(savefile, iteration), "rb") as file:
            Ulist = pickle.load(file)
        with open("{}ngV_iter{}.p".format(savefile, iteration), "rb") as file:
            Vlist = pickle.load(file)
    except (IOError):
        Ulist = Vlist = None

    return Ulist, Vlist


def try_load_UVT(savefile, iteration, time):
    Ulist = Vlist = times = None

    try:
        with open("{}ngU_iter{}_time{}.p".format(savefile, iteration, time), "rb") as file:
            Ulist = pickle.load(file)
        with open("{}ngV_iter{}_time{}.p".format(savefile, iteration, time), "rb") as file:
            Vlist = pickle.load(file)
        with open("{}ngtimes_iter{}_time{}.p".format(savefile, iteration, time), "rb") as file:
            times = pickle.load(file)
    except (IOError):
        Ulist = Vlist = times = None

    return Ulist, Vlist, times


def save_UV(U, V, savefile, iteration):
    with open("{}ngU_iter{}.p".format(savefile, iteration), "wb") as file:
        pickle.dump(Ulist, file, pickle.HIGHEST_PROTOCOL)
    with open("{}ngV_iter{}.p".format(savefile, iteration), "wb") as file:
        pickle.dump(Vlist, file, pickle.HIGHEST_PROTOCOL)


def save_UVT(U, V, times, savefile, iteration, time):
    with open("{}ngU_iter{}_time{}.p".format(savefile, iteration, time), "wb") as file:
        pickle.dump(Ulist, file, pickle.HIGHEST_PROTOCOL)
    with open("{}ngV_iter{}_time{}.p".format(savefile, iteration, time), "wb") as file:
        pickle.dump(Vlist, file, pickle.HIGHEST_PROTOCOL)
    with open("{}ngtimes_iter{}_time{}.p".format(savefile, iteration, time), "wb") as file:
        pickle.dump(times, file, pickle.HIGHEST_PROTOCOL)


def do_train_step(Ulist, Vlist, pmi, b_ind, t, lam, tau, gam, emph, rank):
    for b_num, ind in enumerate(b_ind, 1):  # select a mini batch
        if len(b_ind) > 1:
            print("Batch {}/{} ...".format(b_num, len(b_ind)))

        ## UPDATE V
        # get data
        pmi_seg = pmi[:, ind].todense()

        if t == 0:
            vp = np.zeros((len(ind), rank))
            up = np.zeros((len(ind), rank))
            iflag = True
        else:
            vp = Vlist[t - 1][ind, :]
            up = Ulist[t - 1][ind, :]
            iflag = False

        if t == len(T) - 1:
            vn = np.zeros((len(ind), rank))
            un = np.zeros((len(ind), rank))
            iflag = True
        else:
            vn = Vlist[t + 1][ind, :]
            un = Ulist[t + 1][ind, :]
            iflag = False

        Vlist[t][ind, :] = util.update(
            Ulist[t], emph * pmi_seg, vp, vn, lam, tau, gam, ind, iflag
        )
        Ulist[t][ind, :] = util.update(
            Vlist[t], emph * pmi_seg, up, un, lam, tau, gam, ind, iflag
        )


def do_training(lam, tau, gam, emph, rank, num_iters, batch_size):
    savefile = (
        savehead + "L" + str(lam) + "T" + str(tau) + "G" + str(gam) + "A" + str(emph)
    )

    print("Initializing ...")
    # Ulist, Vlist = util.initvars(num_words, T, r, trainhead)
    Ulist, Vlist = util.import_static_init(T)
    print(Ulist)
    print(Vlist)

    print("Preparing batch indices ...")
    if batch_size < num_words:
        b_ind = util.getbatches(num_words, batch_size)
    else:
        b_ind = [range(num_words)]

    # --------------------------------

    start_time = time.time()

    # sequential updates
    for iteration in range(num_iters):
        print('-' * 78)
        print_params(rank, lam, tau, gam, emph, num_iters)

        # try restoring previous training state
        if SAVEPOINT_ITERATION:
            Ulist2, Vlist2 = try_load_UV(savefile, iteration)
            if Ulist2 and Vlist2:
                print("Iteration {} loaded succesfully.".format(iteration))
                Ulist, Vlist = Ulist2, Vlist2
                continue

        loss = 0  # unused

        # shuffle times  # unused
        times = T if iteration == 0 else np.random.permutation(T)

        for tx, time_ in enumerate(times):  # select next/a time
            print("Iteration {}/{}, Time {}/{}".format(iteration + 1, num_iters, tx + 1, len(times)))

            if SAVEPOINT_ITER_TIME:
                Ulist2, Vlist2, times2 = try_load_UV(savefile, iteration, tx)
                if Ulist2 and Vlist2 and times2:
                    print('Iteration {}, Time {} loaded succesfully'.format(iteration, tx))
                    Ulist, Vlist, times = Ulist2, Vlist2, times2
                    continue

            filename = '{}{}.csv'.format(trainhead, tx)
            # print("Loading current trainings data from: {}".format(filename))
            pmi = util.getmat(filename, num_words, False)

            do_train_step(Ulist, Vlist, pmi, b_ind, tx, lam, tau, gam, emph, rank)

            if SAVEPOINT_ITER_TIME:
                save_UVT(Ulist, Vlist, times, savefile, iteration, tx)

        # save
        print("time elapsed = ", time.time() - start_time)

        if SAVEPOINT_ITERATION:
            save_UV(Ulist, Vlist)


if __name__ == "__main__":
    num_iters = 5  # total passes over the data
    lam = 10  # frob regularizer
    gam = 100  # forcing regularizer
    tau = 50  # smoothing regularizer
    rank = 50  # rank
    batch_size = num_words  # batch size
    emph = 1  # emphasize the nonzero

    args = sys.argv
    for i in range(1, len(args)):
        if args[i] == "-r":
            rank = int(float(args[i + 1]))
        if args[i] == "-iters":
            num_iters = int(float(args[i + 1]))
        if args[i] == "-lam":
            lam = float(args[i + 1])
        if args[i] == "-tau":
            tau = float(args[i + 1])
        if args[i] == "-gam":
            gam = float(args[i + 1])
        if args[i] == "-b":
            batch_size = int(float(args[i + 1]))
        if args[i] == "-emph":
            emph = float(args[i + 1])

    if not (SAVEPOINT_ITERATION or SAVEPOINT_ITER_TIME):
        raise Exception('Should somehow store results ...!')

    print("Starting training with following parameters:")
    print_params(rank, lam, tau, gam, emph, num_iters)
    print("There are a total of {} words and {} time points.".format(num_words, T))

    print("=" * 78)

    do_training(lam, tau, gam, emph, rank, num_iters, batch_size)

    print('Done.')