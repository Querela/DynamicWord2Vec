#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:10:42 2016

"""

# main script for time CD
# trainfile has lines of the form
# tok1,tok2,pmi

import datetime
import os
import sys
import time

import numpy as np
import util_timeCD as util
import pickle as pickle

# PARAMETERS

num_words = 20936  # number of words in vocab (11068100/20936 for ngram/nyt)
T = range(1990, 2016)  # total number of time points (20/range(27) for ngram/nyt)

trainhead = "data/wordPairPMI_"  # location of training data
result_dir = 'results'

SAVEPOINT_ITERATION = True  # save after each iteration (and restore)
SAVEPOINT_ITER_TIME = False  # save after each iteration and each time point (and also restore)
INIT_RANDOM = False  # load static embedding matrix (e. g. previous result) or initialize randomly


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


def save_UV(Ulist, Vlist, savefile, iteration):
    with open("{}ngU_iter{}.p".format(savefile, iteration), "wb") as file:
        pickle.dump(Ulist, file, pickle.HIGHEST_PROTOCOL)
    with open("{}ngV_iter{}.p".format(savefile, iteration), "wb") as file:
        pickle.dump(Vlist, file, pickle.HIGHEST_PROTOCOL)


def save_UVT(Ulist, Vlist, times, savefile, iteration, time):
    with open("{}ngU_iter{}_time{}.p".format(savefile, iteration, time), "wb") as file:
        pickle.dump(Ulist, file, pickle.HIGHEST_PROTOCOL)
    with open("{}ngV_iter{}_time{}.p".format(savefile, iteration, time), "wb") as file:
        pickle.dump(Vlist, file, pickle.HIGHEST_PROTOCOL)
    with open("{}ngtimes_iter{}_time{}.p".format(savefile, iteration, time), "wb") as file:
        pickle.dump(times, file, pickle.HIGHEST_PROTOCOL)


def do_train_step(Ulist, Vlist, pmi, b_ind, t, num_times, lam, tau, gam, emph, rank):
    for b_num, ind in enumerate(b_ind, 1):  # select a mini batch
        if len(b_ind) > 1:
            print("Batch {}/{} ...".format(b_num, len(b_ind)))

        ## UPDATE V
        # get data
        pmi_seg = pmi[:, ind].todense()

        iflag = False  # condition of following may only be true for last one
        if t == 0:
            vp = np.zeros((len(ind), rank))
            up = np.zeros((len(ind), rank))
            iflag = True
        else:
            vp = Vlist[t - 1][ind, :]
            up = Ulist[t - 1][ind, :]

        if t == num_times - 1:
            vn = np.zeros((len(ind), rank))
            un = np.zeros((len(ind), rank))
            iflag = True
        else:
            vn = Vlist[t + 1][ind, :]
            un = Ulist[t + 1][ind, :]

        Vlist[t][ind, :] = util.update(
            Ulist[t], emph * pmi_seg, vp, vn, lam, tau, gam, ind, iflag
        )
        Ulist[t][ind, :] = util.update(
            Vlist[t], emph * pmi_seg, up, un, lam, tau, gam, ind, iflag
        )


def do_training(lam, tau, gam, emph, rank, time_range, num_iters, batch_size):
    savefile = 'L{lam}T{tau}G{gam}A{emph}'.format(lam=lam, tau=tau, gam=gam, emph=emph)
    savefile = os.path.join(result_dir, savefile)

    print("Initializing ...")
    if INIT_RANDOM:
        Ulist, Vlist = util.initvars(num_words, time_range, rank)
    else:
        Ulist, Vlist = util.import_static_init(time_range)
    # print(Ulist)
    # print(Vlist)

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
        # print_params(rank, lam, tau, gam, emph, num_iters)

        # try restoring previous training state
        if SAVEPOINT_ITERATION:
            Ulist2, Vlist2 = try_load_UV(savefile, iteration)
            if Ulist2 and Vlist2:
                print("Iteration {} loaded succesfully.".format(iteration))
                Ulist, Vlist = Ulist2, Vlist2
                continue

        loss = 0  # unused

        # shuffle times  # unused
        times = time_range if iteration == 0 else np.random.permutation(time_range)

        for time_step, time_period in enumerate(times):  # select next/a time
            time_ittm_start = time.time()
            print("Iteration {}/{}, Time {}/{} ({}) ...".format(iteration + 1, num_iters, time_step + 1, len(times), time_period),
                  end='', flush=True)

            if SAVEPOINT_ITER_TIME:
                Ulist2, Vlist2, times2 = try_load_UV(savefile, iteration, time_step)
                if Ulist2 and Vlist2 and times2:
                    print('\nIteration {}, Time {} loaded succesfully'.format(iteration, time_step))
                    Ulist, Vlist, times = Ulist2, Vlist2, times2
                    continue

            filename = '{}{}.csv'.format(trainhead, time_range.index(time_period))
            # print("\nLoading current trainings data (time: {}, at: {}) from: {}".format(time_period, time_range.index(time_period), filename))
            pmi = util.getmat(filename, num_words, False)

            do_train_step(Ulist, Vlist, pmi, b_ind, time_step, len(times), lam, tau, gam, emph, rank)

            if SAVEPOINT_ITER_TIME:
                save_UVT(Ulist, Vlist, times, savefile, iteration, time_step)

            time_ittm_end = time.time()
            print(' {:.2f} sec'.format(time_ittm_end - time_ittm_start))

        # save
        print("Time elapsed = {}".format(datetime.timedelta(seconds=int(time.time() - start_time))))

        if SAVEPOINT_ITERATION:
            save_UV(Ulist, Vlist, savefile, iteration)

    print('Save results to: {}'.format(result_dir))
    sio.savemat("{}/embeddings_Unew.mat".format(result_dir), {"emb": Ulist})
    sio.savemat("{}/embeddings_Vnew.mat".format(result_dir), {"emb": Vlist})


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

    if not os.path.exists(result_dir):
        print('Make result dir: {}'.format(result_dir))
        os.mkdir(result_dir)

    if not (SAVEPOINT_ITERATION or SAVEPOINT_ITER_TIME):
        raise Exception('Should somehow store results ...!')

    print("Starting training with following parameters:")
    print_params(rank, lam, tau, gam, emph, num_iters)
    print("There are a total of {} words and {} time points.".format(num_words, T))

    print("=" * 78)

    do_training(lam, tau, gam, emph, rank, T, num_iters, batch_size)

    print('Done.')