
import numpy as np
import embedding_util
import argparse
import logging

import utils
import constants

def get_embedding(emb, opt=None, noemb="", dim=constants.DEFAULT_EMBEDDING_DIM):
    np_emb = embedding_util.load(emb, strategy=opt, dim=dim)

    logging.debug("Embedding 'noemb' shape: {np_emb.shape}")

    return np_emb

def compare(emb1, opt1, emb2, opt2, dim=constants.DEFAULT_EMBEDDING_DIM):
    np_emb1 = get_embedding(emb1, opt1, noemb="1", dim=dim)
    np_emb2 = get_embedding(emb2, opt2, noemb="2", dim=dim)

    if np_emb1.shape != np_emb2.shape:
        logging.warning("Shapes mismatch")

        return False

    return embedding_util.compare(np_emb1, np_emb2, verbose=verbose)

def identical(emb1, opt1, emb2, opt2, dim=constants.DEFAULT_EMBEDDING_DIM):
    np_emb1 = get_embedding(emb1, opt1, noemb="1", dim=dim)
    np_emb2 = get_embedding(emb2, opt2, noemb="2", dim=dim)

    if np_emb1.shape != np_emb2.shape:
        logging.warning("Shapes mismatch")

        return False

    return (np_emb1 == np_emb2).all()

def main(args):
    emb1 = args.embedding_file_1
    emb2 = args.embedding_file_2
    opt1 = args.embedding_opt_1
    opt2 = args.embedding_opt_2
    dim = args.dim

    result = compare(emb1, opt1, emb2, opt2, dim=dim)

    print(f"Compare: {result}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility to check the loss of precision when applying some embedding optimization')

    # Mandatory
    parser.add_argument('embedding_file_1', metavar='embedding-file-1',
                        help='Path to the 1st embedding')
    parser.add_argument('embedding_file_2', metavar='embedding-file-2',
                        help='Path to the 2nd embedding')
    parser.add_argument('embedding_opt_1', metavar='embedding-opt-1', default=None,
                        help='Embedding optimization which we need to apply in order load the 1st embedding')
    parser.add_argument('embedding_opt_2', metavar='embedding-opt-2', default=None,
                        help='Embedding optimization which we need to apply in order load the 2nd embedding')

    # Other
    parser.add_argument('--dim', type=int, metavar='N', default=constants.DEFAULT_EMBEDDING_DIM,
                        help='Embedding dimensionality')
    parser.add_argument('--logging-level', metavar='N', type=int, default=constants.DEFAULT_LOGGING_LEVEL,
                        help=f'Logging level. Default value is {constants.DEFAULT_LOGGING_LEVEL}')

    args = parser.parse_args()

    utils.set_up_logging(level=args.logging_level)

    main(args)
