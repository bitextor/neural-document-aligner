
import math
import logging

import numpy as np

def levenshtein(embedding_src, embedding_trg, nfactor=1, diff_function_bool=lambda x, y: x != y,
                diff_function_value=lambda x, y: 1.0):
    """Full levenshtein with no optimization applied
    """
    m = np.zeros((len(embedding_src) + 1, len(embedding_trg) + 1))
    insertion_cost = 1.0
    deletion_cost = 1.0

    for row in range(m.shape[0]):
        for col in range(m.shape[1]):
            if row == 0:
                m[row][col] = col * insertion_cost
                continue
            if col == 0:
                m[row][col] = row * deletion_cost
                continue

            diff = 0.0

            if diff_function_bool(embedding_src[row - 1], embedding_trg[col - 1]):
                diff = diff_function_value(embedding_src[row - 1], embedding_trg[col - 1])

            m[row][col] = min(min(m[row][col - 1] + insertion_cost,
                                  m[row - 1][col] + deletion_cost),
                                  m[row - 1][col - 1] + diff)

    return {"matrix": m,
            "value": m[-1][-1],
            "similarity": 1.0 - m[-1][-1] / nfactor}

def levenshtein_opt(embedding_src, embedding_trg, nfactor=1, diff_function_bool=lambda x, y: x != y,
                    diff_function_value=lambda x, y: 1.0):
    """Levenshtein optimization which uses a matrix of only 2*n rows instead of m*n
    """
    m = np.zeros((2, len(embedding_trg) + 1))
    insertion_cost = 1.0
    deletion_cost = 1.0

    for row in range(len(embedding_src) + 1):
        for col in range(m.shape[1]):
            if row == 0:
                m[row][col] = col * insertion_cost
                continue
            if col == 0:
                m[1][col] = row * deletion_cost
                continue

            diff = 0.0

            if diff_function_bool(embedding_src[row - 1], embedding_trg[col - 1]):
                diff = diff_function_value(embedding_src[row - 1], embedding_trg[col - 1])

            m[1][col] = min(min(m[1][col - 1] + insertion_cost,
                                m[1 - 1][col] + deletion_cost),
                                m[1 - 1][col - 1] + diff)
        if row != 0:
            m[0] = m[1]

    return {"matrix": m,
            "value": m[-1][-1],
            "similarity": 1.0 - m[-1][-1] / nfactor}

def levenshtein_opt_space_and_band(embedding_src, embedding_trg, nfactor=1, percentage=0.3, early_stopping=0.05,
                                   diff_function_bool=lambda x, y: x != y, diff_function_value=lambda x, y: 1.0):
    """Levenshtein optimization which uses a matrix of only 2*n rows instead of m*n.
       Besides, only processes the central diagonal of the matrix (central band).
       If the data is not big enough, "levenshtein_opt" will be used instead

       To disable early stopping: early_stopping=np.inf
    """
    if (len(embedding_src) < 10 or len(embedding_trg) < 10):
        return levenshtein_opt(embedding_src, embedding_trg, nfactor=nfactor, diff_function_bool=diff_function_bool,
                               diff_function_value=diff_function_value)

    m = np.ones((2, len(embedding_trg) + 1)) * np.inf
    insertion_cost = 1.0
    deletion_cost = 1.0
    perc_cols = math.floor((len(embedding_src) + 1) * percentage)
    perc_rows = math.floor((len(embedding_trg) + 1) * percentage)

    for i in range(m.shape[1]):
        m[0][i] = i * insertion_cost

    row = 0
    col = 0
    max_rows = len(embedding_src) + 1
    max_cols = m.shape[1]

    while row < max_rows:
        currently_iterated = row / max_rows # [0, 1]
        col = max(0, math.floor(currently_iterated * max_cols) - perc_cols)
        col_limit = min(max_cols, math.floor(currently_iterated * max_cols) + perc_cols)

        if (row and col):
            m[0][:col] = np.inf

        while col < col_limit:
            if row == 0:
                m[row][col] = col * insertion_cost
                col += 1
                continue
            if col == 0:
                m[1][col] = row * deletion_cost
                col += 1
                continue

            diff = 0.0

            if diff_function_bool(embedding_src[row - 1], embedding_trg[col - 1]):
                diff = diff_function_value(embedding_src[row - 1], embedding_trg[col - 1])

            m[1][col] = min(min(m[1][col - 1] + insertion_cost,
                                m[1 - 1][col] + deletion_cost),
                                m[1 - 1][col - 1] + diff)

            col += 1

        if row:
            m[0] = m[1]

            if min(row * deletion_cost, min(m[0])) > early_stopping:
#                logging.debug(f"Early stopping: [{m[0][0]} ... {m[0][-1]}] (row: {row})")

                return {"matrix": m,
                        "value": float(nfactor),
                        "similarity": 0.0}

        row += 1

    return {"matrix": m,
            "value": m[-1][-1],
            "similarity": 1.0 - m[-1][-1] / nfactor}
