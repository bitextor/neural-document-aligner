
import sys
import logging

import numpy as np

DEFAULT_EMBEDDING_DIM = 768

strategy_2_bits = 8
strategy_2_bins = (np.array(range(2 ** strategy_2_bits - 1), dtype=np.float32) - (2 ** strategy_2_bits - 1) // 2) / ((2 ** strategy_2_bits) / 2)
strategy_2_bins_recover = (np.array(range(2 ** strategy_2_bits), dtype=np.float32) - (2 ** strategy_2_bits - 1) // 2) / ((2 ** strategy_2_bits) / 2)

def if_(l, arg, r_ok, r_nok):
    if l(arg):
        return r_ok
    return r_nok

def get_original_embedding_from_optimized(embedding=None, file=None, dim=DEFAULT_EMBEDDING_DIM, strategy=1, to_float32=True):
    if (embedding is None and file is None):
        logging.error("'embedding' or 'file' must have value (if both, 'file' will be used)")
        return None

    x = None

    if file:
        if file == '-':
            x = sys.stdin.buffer.read()

        if strategy == 1:
            if file == '-':
                embedding = np.frombuffer(x, dtype=np.float16)
            else:
                embedding = np.fromfile(file, dtype=np.float16, count=-1)
        elif strategy == 2:
            if file == '-':
                embedding = np.frombuffer(x, dtype=np.uint8)
            else:
                embedding = np.fromfile(file, dtype=np.uint8, count=-1)
        else:
            logging.error(f"Unknown optimization strategy ({strategy}): returning None")
            return None

        x = embedding

    # Sanity check
    if len(embedding.shape) not in (1, 2):
        logging.error(f"Unexpected shape ({embedding.shape}): returning None")
        return None
    if (embedding.shape[len(embedding.shape) - 1] == 0 or embedding.shape[len(embedding.shape) - 1] % dim != 0):
        logging.error(f"Unexpected shape ({embedding.shape}): 0 or not mod {dim}: returning None")
        return None

    if not file:
        x = embedding.copy()

    # Strategies have to work with shape length 1 and 2
    if strategy == 1:
        x = x
    elif strategy == 2:
        # vector quantization (range [-1., 1.])
        x = strategy_2_bins_recover[x]
    else:
        logging.error(f"Unknown optimization strategy: returning None")
        return None

    if to_float32:
        x = x.astype(np.float32)

    return x

def compare(x, y, atol=1.0, rtol=1e-4, verbose=True):
    sub1 = np.abs(x - y)
    sub2 = np.abs(y - x)

    aclose = np.sum((sub1 + sub2) / 2.0)
    rclose = (np.mean(sub1) + np.mean(sub2)) / 2.0
    close = (aclose < atol and rclose / 2 < rtol)

    if verbose:
        if not close:
            logging.warning(f"Too much precision lost due to optimization (rtol of {rtol} vs {rclose}; atol of {atol} vs {aclose})")
        else:
            logging.info(f"{{aclose: {aclose}, rclose: {rclose}}}")

    return close

def load(file, dim=DEFAULT_EMBEDDING_DIM, strategy=None, file_is_fd=False, to_float32=True):
    x = None

    if file_is_fd:
        x = np.load(file)

        if strategy is not None:
            x = get_original_embedding_from_optimized(embedding=x, strategy=strategy, dim=dim, to_float32=to_float32)
    else:
        if strategy is not None:
            x = get_original_embedding_from_optimized(file=file, strategy=strategy, dim=dim, to_float32=to_float32)
        else:
            if file == '-':
                x = sys.stdin.buffer.read()
                x = np.frombuffer(x, dtype=np.float32)
            else:
                x = np.fromfile(file, dtype=np.float32, count=-1) # long array of dim * n, where n is the nosentences

        x.resize(x.shape[0] // dim, dim) # resize to shape=(n,dim)

    return x

def get_optimized_embedding(embedding, strategy=1):
    x = embedding.copy()

    # Apply strategy to optimize
    if strategy == 1:
        x = x.astype(np.float16, copy=False)
    elif strategy == 2:
        # linear quantization (range [-1., 1.])
        x = np.digitize(x, strategy_2_bins).astype(np.uint8)
    else:
        logging.warning(f"Unknown optimization strategy ({strategy}): returning embedding without any optimization strategy applied")

    return x

# The provided embedding must be np.float32
def store(embedding, file, strategy=None):
    x = embedding.copy()

    if (x.dtype != np.float64 and x.dtype != np.float32 and x.dtype != np.float16):
        logging.error(f"Unexpected data type ({x.dtype})")
        return
    if len(x.shape) != 2:
        logging.error(f"Unexpected shape ({x.shape})")
        return

    x.resize(x.shape[0] * x.shape[1])

    if strategy is not None:
        x = get_optimized_embedding(x, strategy=strategy)

    if file == '-':
        sys.stdout.buffer.write(x.tobytes())
    else:
        x.tofile(file)

def test_precision(embedding, strategy, dim=DEFAULT_EMBEDDING_DIM, return_optimized_embedding=False):
    if len(embedding.shape) != 2:
        logging.error(f"Unexpected shape ({embedding.shape})")
        return None

    x = embedding.copy()

    x = get_optimized_embedding(x, strategy=strategy)
    x.resize(x.shape[0] * x.shape[1])
    x = get_original_embedding_from_optimized(x, strategy=strategy)
    x.resize(x.shape[0] // dim, dim)

    acc_rerr = np.sum(np.abs(x - embedding))

    logging.info(f"Accumulated relative error (lost precision): {acc_rerr} (avg: {acc_rerr / (x.shape[0] * x.shape[1])})")

    if return_optimized_embedding:
        return compare(x, embedding), x
    else:
        return compare(x, embedding)
