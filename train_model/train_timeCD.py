#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main script for time CD

trainfile has lines of the form:
    tok1,tok2,pmi

Created on Thu Nov 10 13:10:42 2016
"""

import ast
import os
import time

import numpy as np

import util_timeCD as util
from util_shared import set_base_indent_level, iprint, get_time_diff


# ----------------------------------------------------------------------------


def print_train_params(rank, lam, tau, gam, emph, num_iterations):
    """Dump parameter values.

    :param rank: rank/dimension of embeddings (?)
    :param lam: frob regularizer
    :param tau: smoothing regularizer / time regularizer
    :param gam: forcing regularizer / symmetry regularizer
    :param emph: emphasize the nonzero
    :param num_iterations: number of iterations over data

    """
    iprint("~ rank = {}".format(rank))
    iprint("~ frob regularizer = {}".format(lam))
    iprint("~ time regularizer = {}".format(tau))
    iprint("~ symmetry regularizer = {}".format(gam))
    iprint("~ emphasize param = {}".format(emph))
    iprint("~ total iterations = {}".format(num_iterations))


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
    randomize_times=False,
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
    :param batch_size: size for batching (Default value = None)
    :param data_file: if given a file with initial embeddings (Default value = None)
    :param randomize_times: Randomize timepoints in timerange while training (Default value = False)
    :param savepoint_iteration: store current training results per iteration and try to retore from there (Default value = True)
    :param savepoint_iter_time: store current training results per iteration and timepoint and try to restore there (Default value = False)

    """
    set_base_indent_level()

    savefile = "L{lam}T{tau}G{gam}A{emph}".format(lam=lam, tau=tau, gam=gam, emph=emph)
    savefile = os.path.join(result_dir, savefile)

    # add inclusive end timepoint ...
    # TODO: think of something better ...
    time_range = range(time_range[0], time_range[-1] + 2)

    iprint("* Initializing ...")
    if data_file is None:
        Ulist, Vlist = util.init_emb_random(num_words, time_range, rank)
    else:
        Ulist, Vlist = util.init_emb_static(data_file, time_range)
    # print(Ulist)
    # print(Vlist)

    iprint("* Preparing batch indices ...")
    if batch_size is not None and batch_size < num_words:
        b_ind = util.make_batches(num_words, batch_size)
    else:
        b_ind = [range(num_words)]

    # --------------------------------

    start_time = time.time()

    # sequential updates
    for iteration in range(num_iters):
        iprint("-" * 78)
        # print_train_params(rank, lam, tau, gam, emph, num_iters)

        # try restoring previous training state
        if savepoint_iteration:
            Ulist2, Vlist2 = util.try_load_UV(savefile, iteration)
            if Ulist2 and Vlist2:
                iprint("* Iteration {} loaded succesfully.".format(iteration), level=1)
                Ulist, Vlist = Ulist2, Vlist2
                continue

        loss = 0  # unused

        times = time_range
        # shuffle times
        if randomize_times:
            # TODO: keep times for even iterations un-randomized?
            if iteration > 0 and iteration < (num_iters - 1):
                times = np.random.permutation(time_range)

        for time_step, time_period in enumerate(times):  # select next/a time
            time_ittm_start = time.time()
            iprint(
                "* Iteration {} ({}/{}), Time {} ({}/{}) ...".format(
                    iteration,
                    iteration + 1,
                    num_iters,
                    time_period,
                    time_step + 1,
                    len(times),
                ),
                end="",
                flush=True,
                level=1,
            )

            if savepoint_iter_time:
                Ulist2, Vlist2, times2 = util.try_load_UV(
                    savefile, iteration, time_step
                )
                if Ulist2 and Vlist2 and times2:
                    iprint(
                        "* Iteration {}, Time {} loaded succesfully".format(
                            iteration, time_step
                        ),
                        level=1,
                    )
                    Ulist, Vlist, times = Ulist2, Vlist2, times2
                    continue

            pmi = util.load_train_data(data_dir, num_words, time_range, time_period)

            util.do_train_step(
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
                util.save_UVT(Ulist, Vlist, times, savefile, iteration, time_step)

            time_ittm_end = time.time()
            print(" {:.2f} sec".format(time_ittm_end - time_ittm_start))

        print("* Total time elapsed = {}".format(get_time_diff(start_time)))

        # save
        if savepoint_iteration:
            util.save_UV(Ulist, Vlist, savefile, iteration)

    iprint("* Save results to: {}".format(result_dir))
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
        "--randomize-timepoints",
        action="store_true",
        help="randomize timepoints in timerange while training",
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
        iprint("! Default to default value for time_range, {}".format(ex))
        args.time_range = range(*time_range)

    if args.batch_size <= 0:
        args.batch_size = args.num_words

    return args


if __name__ == "__main__":
    #: parse arguments, use defaults
    set_base_indent_level()
    args = parse_args()

    #: warn if no savepoints
    if not (args.save_per_iteration or args.save_per_iteration_time):
        raise Exception("Should somehow store intermediate results ...!")

    # make results dir
    if not os.path.exists(args.result_dir):
        iprint("! Make results dir: {}".format(args.result_dir))
        os.mkdir(args.result_dir)

    # dump parameters
    iprint("* Starting training with following parameters:")
    print_train_params(args.rank, args.lam, args.tau, args.gam, args.emph, args.iters)
    iprint(
        "* There are a total of {} words and {} time points.".format(
            args.num_words, args.time_range
        )
    )

    iprint("=" * 78)
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
        randomize_times=args.randomize_timepoints,
        savepoint_iteration=args.save_per_iteration,
        savepoint_iter_time=args.save_per_iteration_time,
    )

    iprint("* Done.")
