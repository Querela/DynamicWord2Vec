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
import time

import numpy as np
import util_timeCD as util
import pickle as pickle

# PARAMETERS

num_words = 20936  # number of words in vocab (11068100/20936 for ngram/nyt)
T = range(1990, 2016)  # total number of time points (20/range(27) for ngram/nyt)

trainhead = "data/wordPairPMI_"  # location of training data


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
        with open(
            "{}ngU_iter{}_time{}.p".format(savefile, iteration, time), "rb"
        ) as file:
            Ulist = pickle.load(file)
        with open(
            "{}ngV_iter{}_time{}.p".format(savefile, iteration, time), "rb"
        ) as file:
            Vlist = pickle.load(file)
        with open(
            "{}ngtimes_iter{}_time{}.p".format(savefile, iteration, time), "rb"
        ) as file:
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
    with open(
        "{}ngtimes_iter{}_time{}.p".format(savefile, iteration, time), "wb"
    ) as file:
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


def do_training(
    lam,
    tau,
    gam,
    emph,
    rank,
    time_range,
    num_iters,
    batch_size,
    result_dir,
    data_file=None,
    savepoint_iteration=True,
    savepoint_iter_time=False,
):
    savefile = "L{lam}T{tau}G{gam}A{emph}".format(lam=lam, tau=tau, gam=gam, emph=emph)
    savefile = os.path.join(result_dir, savefile)

    print("Initializing ...")
    if data_file is None:
        Ulist, Vlist = util.initvars(num_words, time_range, rank)
    else:
        Ulist, Vlist = util.import_static_init(data_file, time_range)
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
        print("-" * 78)
        # print_params(rank, lam, tau, gam, emph, num_iters)

        # try restoring previous training state
        if savepoint_iteration:
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
            print(
                "Iteration {}/{}, Time {}/{} ({}) ...".format(
                    iteration + 1, num_iters, time_step + 1, len(times), time_period
                ),
                end="",
                flush=True,
            )

            if savepoint_iter_time:
                Ulist2, Vlist2, times2 = try_load_UV(savefile, iteration, time_step)
                if Ulist2 and Vlist2 and times2:
                    print(
                        "\nIteration {}, Time {} loaded succesfully".format(
                            iteration, time_step
                        )
                    )
                    Ulist, Vlist, times = Ulist2, Vlist2, times2
                    continue

            filename = "{}{}.csv".format(trainhead, time_range.index(time_period))
            # print("\nLoading current trainings data (time: {}, at: {}) from: {}".format(time_period, time_range.index(time_period), filename))
            pmi = util.getmat(filename, num_words, False)

            do_train_step(
                Ulist,
                Vlist,
                pmi,
                b_ind,
                time_step,
                len(times),
                lam,
                tau,
                gam,
                emph,
                rank,
            )

            if savepoint_iter_time:
                save_UVT(Ulist, Vlist, times, savefile, iteration, time_step)

            time_ittm_end = time.time()
            print(" {:.2f} sec".format(time_ittm_end - time_ittm_start))

        print(
            "Total time elapsed = {}".format(
                datetime.timedelta(seconds=int(time.time() - start_time))
            )
        )

        # save
        if savepoint_iteration:
            save_UV(Ulist, Vlist, savefile, iteration)

    print("Save results to: {}".format(result_dir))
    util.save_embeddings("{}/embeddings_Unew.mat".format(result_dir), Ulist)
    util.save_embeddings("{}/embeddings_Vnew.mat".format(result_dir), Vlist)


def parse_args():
    import argparse

    # Default arguments:
    num_iters = 5  # total passes over the data
    lam = 10.0  # frob regularizer
    gam = 100.0  # forcing regularizer
    tau = 50.0  # smoothing regularizer
    rank = 50  # rank
    batch_size = num_words  # batch size
    emph = 1.0  # emphasize the nonzero
    data_file = "data/emb_static.mat"
    result_dir = "results"

    # Parse arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rank", type=float, default=rank, help="rank")
    parser.add_argument(
        "--iters", type=int, default=num_iters, help="iterations over data"
    )
    parser.add_argument("--lam", type=float, default=lam, help="frob regularizer")
    parser.add_argument(
        "--tau",
        type=float,
        default=tau,
        help="smoothing regularizer / time regularizer",
    )
    parser.add_argument(
        "--gam",
        type=float,
        default=gam,
        help="forcing regularizer / symmetry regularizer",
    )
    parser.add_argument(
        "--emph", type=float, default=emph, help="emphasize the nonzero"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=batch_size, help="Batch size"
    )
    parser.add_argument(
        "--init-weights-file",
        type=str,
        default=data_file,
        help="file with initial static weights; if missing then random initialization",
    )
    parser.add_argument(
        "--init-random-weights",
        action="store_true",
        help="initialize with random weights or load static embedding matrix (e. g. previous result)",
    )
    parser.add_argument(
        "--result-dir",
        default=result_dir,
        help="Folder with result and intermediate training files, default: {}".format(
            result_dir
        ),
    )
    parser.add_argument(
        "--save-per-iteration",
        action="store_true",
        default=True,
        help="store results per iteration, will use existing results to skip finished iterations, "
        "always enabled ...",
    )
    parser.add_argument(
        "--save-per-iteration-time",
        action="store_true",
        default=False,
        help="store results per iteration and time",
    )
    args = parser.parse_args()

    if args.init_random_weights:
        args.init_weights_file = None

    return args


if __name__ == "__main__":
    args = parse_args()

    if not (args.save_per_iteration or args.save_per_iteration_time):
        raise Exception("Should somehow store intermediate results ...!")

    if not os.path.exists(args.result_dir):
        print("Make results dir: {}".format(args.result_dir))
        os.mkdir(args.result_dir)

    print("Starting training with following parameters:")
    print_params(args.rank, args.lam, args.tau, args.gam, args.emph, args.iters)
    print("There are a total of {} words and {} time points.".format(num_words, T))

    print("=" * 78)

    do_training(
        args.lam,
        args.tau,
        args.gam,
        args.emph,
        args.rank,
        T,
        args.iters,
        args.batch_size,
        args.result_dir,
        data_file=args.init_weights_file,
        savepoint_iteration=args.save_per_iteration,
        savepoint_iter_time=args.save_per_iteration_time,
    )

    print("Done.")
