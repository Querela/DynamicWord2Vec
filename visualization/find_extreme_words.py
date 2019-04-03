# -*- coding: utf-8 -*-

import ast
import inspect
import os

import numpy as np
import scipy.io as sio

from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------

SKIP_LEVEL = 3


def set_base_indent_level():
    """Set base indentation stack level for `iprint()?  calls."""
    global SKIP_LEVEL

    level = 0

    frame = inspect.currentframe()
    while frame.f_back:
        level += 1
        frame = frame.f_back

    # print('Computed level: {}, result:{}'.format(level, level + 1))

    SKIP_LEVEL = level + 1


def get_indent(skip, more_level=0):
    """Get indentation string.

    :param skip: skip some levels (Default value = SKIP_LEVEL)
    :param more_level: indent more (Default value = 0)

    """
    level = 0

    frame = inspect.currentframe()
    while frame.f_back:
        level += 1
        frame = frame.f_back

    if skip > 0:
        level -= min(max(0, skip), level)
    if more_level > 0:
        level += more_level

    indent = "  " * level
    return indent


def iprint(msg, level=0, *args, **kwargs):
    """Write indented.

    :param msg: strint to write
    :param level: indent more (Default value = 0)
    :param *args: print args
    :param **kwargs: print kwargs

    """
    print("{}{}".format(get_indent(SKIP_LEVEL, more_level=level), msg), *args, **kwargs)


# ---------------------------------------------------------------------------


def compute_emb_norms(emb_all, time_range):
    """Compute embedding vector norms per timepoint and also normalize a copy of the embeddings.

    :param emb_all: embeddings dictionary for each timepoint (Key: "U_%d")
    :param time_range: range of timepoints
    :returns: normalized embeddings, norms per timepoint

    """
    emb_norms = dict()
    norm_all = list()

    # https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/utils/extmath.py#L63
    # https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/preprocessing/data.py#L1513

    for tx, time_point in enumerate(time_range):
        key = "U_{}".format(tx)
        emb = emb_all[key]  # embedding matrix

        # norms = np.linalg.norm(emb, axis=1)  # row norms
        # norms_nonzero = norms[norms == 0.0] = 1.0
        # ...
        emb_norm, norms = normalize(emb, return_norm=True)

        emb_norms[key] = emb_norm
        norm_all = norms

    # re-scale norms
    norm_all = np.array(norm_all)
    norm_all /= np.sum(norm_all, axis=0)  # TODO: scaling between [0,1] ?

    return emb_norms, norm_all


# ---------------------------------------------------------------------------


def find_extremes(
    emb_all,
    time_range,
    words,
    num=10,
    filter_years=None,
    inverse=False,
    add_dists=False,
):
    """Finds words with extreme sum of vector differences between the years.
    With param `filter_years` you can filter for first-last year instead of over all years.

    :param emb_all: embeddings dictionary
    :param time_range: range of timepoints
    :param words: list of words (position is id)
    :param num: number of words to return, if negative then all (Default value = 10)
    :param filter_years: filter for only those years in `time_range` (Default value = None)
    :param inverse: inverse sorting (return un-extremes) (Default value = False)
    :param add_dists: return tuple of word, distance instead of only word (Default value = False)
    :returns: list of words (if `add_dists` then tuple of word and distance)

    """
    if num == 0:
        return list()
    elif num < 0:
        num = len(words)

    diffs_all = list()  # list of diffs per vector
    # diffs_all = np.zeros((len(words), len(time_range) - 1))
    # iprint("? diffs_all: {}".format(diffs_all.shape))

    years = time_range
    if filter_years is not None:
        years = filter_years

    # compute differences between vectors between each year
    last_emb = None
    for year in years:
        key = "U_{}".format(time_range.index(year))
        emb = emb_all[key]  # embedding matrix
        # iprint("? emb: {}".format(emb.shape))

        if last_emb is None:
            last_emb = emb
            continue

        diffs = np.linalg.norm(last_emb - emb, axis=1)
        # iprint("? diffs: {}".format(diffs.shape))
        # diffs_all[:, tx - 1] = diffs  # TODO: assignment does not work???
        diffs_all.append(diffs)

    last_emb = None
    diffs_all = np.array(diffs_all)
    # iprint("? diffs_all: {}".format(diffs_all.shape))
    diffs_all = diffs_all.T
    # iprint("? diffs_all: {}".format(diffs_all.shape))

    # compute sum of distances between years
    dists_all = np.sum(np.absolute(diffs_all), axis=1)
    # iprint("? dists_all: {}".format(dists_all.shape))

    # sort
    if not inverse:
        dists_all *= -1.0
    sort_inds = np.argsort(dists_all)
    inds = sort_inds[:num]

    # get words
    words = [words[i] for i in inds]

    if add_dists:
        sort_dist = np.absolute(dists_all[sort_inds][:num])
        words = list(zip(words, sort_dist))
    # iprint("? Words: {}".format(list(zip(words, sort_dist))))

    return words


# ---------------------------------------------------------------------------


def main(embeddings_filename, time_range, words_file, result_dir):
    """Load data and compute things (distances).

    :param embeddings_filename: file with embeddings
    :param time_range: range of timepoints
    :param words_file: file with word list
    :param result_dir: output directory for results

    """
    set_base_indent_level()

    emb_all = sio.loadmat(embeddings_filename)
    iprint("? emb_all.keys(): {}".format(emb_all.keys()))
    with open(words_file, "r", encoding="utf-8") as fin:
        words = [w.strip() for w in fin]

    iprint("* Compute norms ...")
    emb_norms, norm_all = compute_emb_norms(emb_all, time_range)

    # TODO: compute distances with projected 2D trajectory?
    # may need to TSNE for each word?

    iprint("* Find extreme words ...")
    extremes = find_extremes(emb_norms, time_range, words, num=10)
    iprint("# Extremes: {}".format(extremes))
    unextremes = find_extremes(emb_norms, time_range, words, num=10, inverse=True)
    iprint("# Un-Extremes: {}".format(unextremes))

    iprint("* Find extreme words first-last ...")
    extremes_fl = find_extremes(
        emb_norms,
        time_range,
        words,
        filter_years=(time_range[0], time_range[-1]),
        num=10,
    )
    iprint("# Extremes: {}".format(extremes_fl))
    unextremes_fl = find_extremes(
        emb_norms,
        time_range,
        words,
        filter_years=(time_range[0], time_range[-1]),
        num=10,
        inverse=True,
    )
    iprint("# Un-Extremes: {}".format(unextremes_fl))


def parse_args():
    """Parse arguments. (Has defaults.)

    :returns: Parsed (final) arguments.

    """
    embeddings_filename = "results/embeddings.mat"
    words_file = "data/wordlist.txt"
    result_dir = "results"
    time_range = (1990, 2009)  # 2015)  # range, total number of time points

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--emb-file",
        default=embeddings_filename,
        help="filename with embeddings, default: {}".format(embeddings_filename),
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
        "-w",
        "--words-file",
        default=words_file,
        help="input filename with list of words, default: {}".format(words_file),
    )
    parser.add_argument(
        "--result-dir",
        default=result_dir,
        help="Folder with result and intermediate training files, default: {}".format(
            result_dir
        ),
    )

    args = parser.parse_args()

    try:
        time_range2 = range(*ast.literal_eval(args.time_range))
        args.time_range = time_range2
    except Exception as ex:
        print("! Default to default value for time_range, {}".format(ex))
        args.time_range = range(*time_range)

    args.time_range = range(args.time_range[0], args.time_range[-1] + 2)

    return args


if __name__ == "__main__":
    args = parse_args()

    # make results dir
    if not os.path.exists(args.result_dir):
        print("Make results dir: {}".format(args.result_dir))
        os.mkdir(args.result_dir)

    main(args.emb_file, args.time_range, args.words_file, args.result_dir)
