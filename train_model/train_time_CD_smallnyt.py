#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main script for time CD

trainfile has lines of the form:
    tok1,tok2,pmi

Created on Thu Nov 10 13:10:42 2016
"""

import ast
import datetime
import os
import time

import numpy as np
import util_timeCD as util
import pickle as pickle


def print_params(rank, lam, tau, gam, emph, num_iterations):
    """Dump parameter values.

    :param rank: rank/dimension of embeddings (?)
    :param lam: frob regularizer
    :param tau: smoothing regularizer / time regularizer
    :param gam: forcing regularizer / symmetry regularizer
    :param emph: emphasize the nonzero
    :param num_iterations: number of iterations over data

    """
    print("rank = {}".format(rank))
    print("frob regularizer = {}".format(lam))
    print("time regularizer = {}".format(tau))
    print("symmetry regularizer = {}".format(gam))
    print("emphasize param = {}".format(emph))
    print("total iterations = {}".format(num_iterations))


def try_load_UV(savefile, iteration):
    """Try to load a savepoint of U and V for a given iteration.

    :param savefile: filename prefix, given by parameters
    :param iteration: iteration
    :returns: U, V if able to load, else None, None on error

    """
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
    """Try to load a savepoint of U and V and times for a given iteration and timepoint combination.

    :param savefile: filename prefix, given by parameters
    :param iteration: iteration
    :param time: timepoint in iteration
    :returns: U, V, times if able to load, else None, None, None on error

    """
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
    """Saves embeddings U, V for a given iteration.

    :param Ulist: embedding U
    :param Vlist: embedding V
    :param savefile: filename prefix, given by parameters
    :param iteration: iteration

    """
    with open("{}ngU_iter{}.p".format(savefile, iteration), "wb") as file:
        pickle.dump(Ulist, file, pickle.HIGHEST_PROTOCOL)
    with open("{}ngV_iter{}.p".format(savefile, iteration), "wb") as file:
        pickle.dump(Vlist, file, pickle.HIGHEST_PROTOCOL)


def save_UVT(Ulist, Vlist, times, savefile, iteration, time):
    """Saves embeddings U, V and times for a given iteration.

    :param Ulist: embeddings U
    :param Vlist: embeddings V
    :param times: times (may be randomized)
    :param savefile: filename prefix, given by parameters
    :param iteration: iteration
    :param time: timepoint in iteration

    """
    with open("{}ngU_iter{}_time{}.p".format(savefile, iteration, time), "wb") as file:
        pickle.dump(Ulist, file, pickle.HIGHEST_PROTOCOL)
    with open("{}ngV_iter{}_time{}.p".format(savefile, iteration, time), "wb") as file:
        pickle.dump(Vlist, file, pickle.HIGHEST_PROTOCOL)
    with open(
        "{}ngtimes_iter{}_time{}.p".format(savefile, iteration, time), "wb"
    ) as file:
        pickle.dump(times, file, pickle.HIGHEST_PROTOCOL)


def load_train_data(data_dir, num_words, time_range, time_period):
    """Really not very generic method to load train data PMI file for a given timepoint.

    :param data_dir: directory of data files, PMI word pairs
    :param num_words: number of words in vocabulary
    :param time_range: range object of times
    :param time_period: timepoint (year?)

    """
    filename = "wordPairPMI_{}.csv".format(time_range.index(time_period))
    if data_dir:
        filename = os.path.join(data_dir, filename)
    # print("\nLoading current trainings data (time: {}, at: {}) from: {}".format(time_period, time_range.index(time_period), filename))

    pmi = util.getmat(filename, num_words, False)

    return pmi


def do_train_step(Ulist, Vlist, pmi, b_ind, t, num_times, lam, tau, gam, emph, rank):
    """Do a single training step for a single iteration and timepoint combination.
    Uses b_ind (batching indices) to batch-wise update the whole embedding matrices.

    :param Ulist: embeddings U
    :param Vlist: embeddings V
    :param pmi: PMI word matrix
    :param b_ind: batching indices (list of ranges())
    :param t: current timepoint
    :param num_times: number of timepoints total
    :param lam: frob regularizer
    :param tau: smoothing regularizer / time regularizer
    :param gam: forcing regularizer / symmetry regularizer
    :param emph: emphasize the nonzero
    :param rank: rank/dimension of embeddings (?)

    """
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
    num_words,
    result_dir,
    data_dir,
    batch_size=None,
    data_file=None,
    savepoint_iteration=True,
    savepoint_iter_time=False,
):
    """Do a complete training.
    Able to save current training state per iteration and timepoint and restore
    training from there.

    :param lam: frob regularizer
    :param tau: smoothing regularizer / time regularizer
    :param gam: forcing regularizer / symmetry regularizer
    :param emph: emphasize the nonzero
    :param rank: ranke/dimension of embeddings
    :param time_range: range of time points
    :param num_iters: number of training iterations
    :param num_words: number of words in vocabulary
    :param result_dir: folder to store savepoints and final results in
    :param data_dir: folder with trainings data (PMI word pairs)
    :param batch_size: size for batching
    :param data_file: if given a file with initial embeddings (Default value = None)
    :param savepoint_iteration: store current training results per iteration and try to retore from there (Default value = True)
    :param savepoint_iter_time: store current training results per iteration and timepoint and try to restore there (Default value = False)

    """
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
    if batch_size is not None and batch_size < num_words:
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

            pmi = load_train_data(data_dir, num_words, time_range, time_period)

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
    util.save_embeddings_split(
        "{}/embeddings.mat".format(result_dir), Ulist, time_range
    )


def parse_args():
    """Parse training parameter from commandline args.
    Set defaults if not given.

    :returns: parameters

    """
    import argparse

    # Default arguments:
    num_iters = 5  # total passes over the data
    lam = 10.0  # frob regularizer
    gam = 100.0  # forcing regularizer
    tau = 50.0  # smoothing regularizer
    rank = 50  # rank
    num_words = 20936  # number of words in vocab (11068100/20936 for ngram/nyt)
    batch_size = -1  # batch size, -1 for whole
    emph = 1.0  # emphasize the nonzero
    data_file = "data/emb_static.mat"
    result_dir = "results"
    data_dir = "data"
    time_range = (
        1990,
        2016,
    )  # range, total number of time points (20/range(27) for ngram/nyt)

    # Parse arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rank", type=int, default=rank, help="rank")
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
        "-n",
        "--num-words",
        type=int,
        default=num_words,
        help="number of words in vocabulary",
    )
    parser.add_argument(
        "--time-range",
        type=str,
        default=str(time_range),
        help='time range (years?), format: "year_start,year_end" default: {}'.format(
            str(time_range)
        ),
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=batch_size,
        help="Batch size, -1 for no batches",
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
        "--data-dir",
        default=data_dir,
        help="Folder with data files, PMI word pairs, wordlists etc., default: {}".format(
            data_dir
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

    try:
        time_range2 = range(*ast.literal_eval(args.time_range))
        args.time_range = time_range2
    except Exception as ex:
        print("! Default to default value for time_range, {}".format(ex))
        args.time_range = range(*time_range)

    if args.batch_size <= 0:
        args.batch_size = args.num_words

    return args


if __name__ == "__main__":
    #: parse arguments, use defaults
    args = parse_args()

    #: warn if no savepoints
    if not (args.save_per_iteration or args.save_per_iteration_time):
        raise Exception("Should somehow store intermediate results ...!")

    # make results dir
    if not os.path.exists(args.result_dir):
        print("Make results dir: {}".format(args.result_dir))
        os.mkdir(args.result_dir)

    # dump parameters
    print("Starting training with following parameters:")
    print_params(args.rank, args.lam, args.tau, args.gam, args.emph, args.iters)
    print(
        "There are a total of {} words and {} time points.".format(
            args.num_words, args.time_range
        )
    )

    print("=" * 78)
    # print(args)

    # train
    do_training(
        args.lam,
        args.tau,
        args.gam,
        args.emph,
        args.rank,
        args.time_range,
        args.iters,
        args.num_words,
        args.result_dir,
        args.data_dir,
        batch_size=args.batch_size,
        data_file=args.init_weights_file,
        savepoint_iteration=args.save_per_iteration,
        savepoint_iter_time=args.save_per_iteration_time,
    )

    print("Done.")
