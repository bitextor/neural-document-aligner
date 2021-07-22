
import os
import sys
import logging
import datetime

import constants

def expand_and_real_path_and_exists(path, rtn_path_if_doesnt_exist=False, func_check_exists=os.path.isfile,
                                    raise_exception=False, exception=FileNotFoundError):
    """Expand the provided path with real path and user information and handle exceptions if necessary
       based on the existence
    """
    p = os.path.realpath(os.path.expanduser(path))

    if not func_check_exists(p):
        if raise_exception:
            raise exception(p)

        if rtn_path_if_doesnt_exist:
            return path

    return p

def get_current_datetime_filename(format="%Y%m%d%H%M%S"):
    return datetime.datetime.now().strftime(format)

def print_full_numpy_array(arr, err=True):
    """It prints a whole numpy array instead of just printing a few elements
    """
    numpy_threshold = np.get_printoptions()["threshold"]
    np.set_printoptions(threshold=sys.maxsize)

    if err:
        sys.stderr.write(f"{arr}\n")
    else:
        print(arr)

    np.set_printoptions(threshold=numpy_threshold)

def get_nolines(path):
    """Return the number of lines of a file
    """
    nolines = 0

    with open(path, "r") as f:
        for _ in f:
            nolines += 1

    return nolines

def set_up_logging(filename=None, level=logging.INFO, format=constants.DEFAULT_LOGGING_FORMAT):
    """It sets up the logging library
    """
    handlers = [
        logging.StreamHandler()
    ]

    if filename is not None:
        handlers.append(logging.FileHandler(filename))

    logging.basicConfig(handlers=handlers, level=level,
                        format=format)
