# -*- coding: utf-8 -*-

import datetime
import inspect
import time

# ---------------------------------------------------------------------------
# - shared constants / defaults


class defaults:
    embeddings_filename = "embeddings.mat"
    wordlist_file = "wordlist.txt"
    data_dir = "data"
    result_dir = "results"


# ---------------------------------------------------------------------------

SKIP_LEVEL = 3


def set_base_indent_level(delta=0):
    """Set base indentation stack level for `iprint()?  calls.

    :param delta: relative change applied to final level (Default value = 0)

    """
    global SKIP_LEVEL

    level = 0

    frame = inspect.currentframe()
    while frame.f_back:
        level += 1
        frame = frame.f_back

    # print('Computed level: {}, result:{}'.format(level, level + 1))

    SKIP_LEVEL = level + 1
    SKIP_LEVEL += delta


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


def get_time_diff(start_time, end_time=None):
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


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
