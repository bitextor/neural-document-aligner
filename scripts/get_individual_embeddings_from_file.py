#!/usr/bin/env python3

import os
import sys
import logging
import argparse

import numpy as np

sys.path.append(f"{__file__.rsplit('/', 1)[0]}/../neural_document_aligner")

import utils.utils as utils
import utils.embedding_utils as embedding_utils
import constants
import exceptions

def load_embeddings(embeddings_path, dim=constants.DEFAULT_EMBEDDING_DIM, optimization_strategy=None):
    embeddings = []
    total_sent_level_embs = 0
    first_debug_message = False

    with open(embeddings_path, "rb") as f:
        try:
            while True:
                embedding = np.load(f)

                if not first_debug_message:
                    first_debug_message = True

                    logging.debug(f"First loaded embedding dtype: {embedding.dtype}")
                    logging.debug(f"First loaded embedding shape: {embedding.shape}")

                if optimization_strategy is not None:
                    embedding = embedding_utils.get_original_embedding_from_optimized(embedding=embedding, dim=dim, strategy=optimization_strategy)

                total_sent_level_embs += embedding.shape[0]

                embeddings.append(embedding)
        except ValueError:
            pass

    logging.debug(f"Total sentence-level embeddings: {total_sent_level_embs}")

    return embeddings

def store_embeddings(output_dir, embeddings, prefix, optimization_strategy=None):
    first_debug_message = False

    for idx, embedding in enumerate(embeddings):
        output = f"{output_dir}/{prefix}.{idx}.embedding"
        modified = False

        if optimization_strategy is not None:
            embedding = embedding_utils.get_optimized_embedding(embedding, strategy=optimization_strategy)

        if not first_debug_message:
            first_debug_message = True

            logging.debug(f"First stored embedding dtype: {embedding.dtype}")
            logging.debug(f"First stored embedding shape: {embedding.shape}")


        while os.path.isfile(output):
            modified = True
            output = f"{output}.pad"

        if modified:
            logging.warning(f"Embedding #{idx} name has been padded since could not store the embedding with the original name: {output}")

        embedding.tofile(output)

def main(args):
    input_file = args.input_file
    output_dir = args.output_dir
    prefix = args.prefix
    dim = args.dim
    optimization_strategy = None if args.optimization_strategy == 0 else args.optimization_strategy
    store_optimization_strategy = None if args.store_optimization_strategy == 0 else args.store_optimization_strategy

    if not os.path.isfile(input_file):
        raise FileNotFoundError(input_file)
    if not os.path.isdir(output_dir):
        raise exceptions.DirNotFoundError(output_dir)

    if optimization_strategy is not None:
        logging.info(f"Loading embeddings with optimization strategy: {optimization_strategy}")
    if store_optimization_strategy is not None:
        logging.info(f"Storing embeddings with optimization strategy: {store_optimization_strategy}")

    embeddings = load_embeddings(input_file, dim=dim, optimization_strategy=optimization_strategy)

    logging.info(f"Loaded embeddings: {len(embeddings)}")

    store_embeddings(output_dir, embeddings, prefix, optimization_strategy=store_optimization_strategy)

    logging.info(f"Embeddings have been stored: '{output_dir}/{prefix}*'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform merged embedding file to individual sentence-embedding files')

    # Embedding
    parser.add_argument('input_file', metavar='input-file',
        help='Embedding merged input file')
    parser.add_argument('output_dir', metavar='output-dir',
        help='Output direcotry where the embeddings will be stored')

    # Optional
    parser.add_argument('--prefix', default="embedding_doc", metavar='STR',
        help=f'Prefix of the documents which will be stored in output_dir. The default value is \'embedding_doc\'')
    parser.add_argument('--dim', default=constants.DEFAULT_EMBEDDING_DIM, type=int, metavar='N',
        help=f'Dimensionality of the provided embeddings. The default value is {constants.DEFAULT_EMBEDDING_DIM}')
    parser.add_argument('--optimization-strategy', default=None, type=int,
        choices=range(0, 2 + 1),
        help='Optimization strategy of the embeddings which are going to be loaded. The default value is do not apply any strategy')
    parser.add_argument('--store-optimization-strategy', default=None, type=int,
        choices=range(0, 2 + 1),
        help='Optimization strategy of the embeddings which are going to be stored. The default value is do not apply any strategy')
    parser.add_argument('--logging-level', metavar='N', type=int, default=constants.DEFAULT_LOGGING_LEVEL,
                        help=f'Logging level. Default value is {constants.DEFAULT_LOGGING_LEVEL}')

    args = parser.parse_args()

    utils.set_up_logging(level=args.logging_level)

    main(args)
