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

def process_input_file(input_file):
    embeddings_path = []

    with open(input_file) as f:
        for idx, line in enumerate(f):
            line = line.strip()

            if not os.path.isfile(line):
                logging.warning(f"Embedding file does not exist in line #{idx + 1} (it will be skipped)")
                continue

            embeddings_path.append(line)

    return embeddings_path

def load_embeddings(embeddings_path, dim=constants.DEFAULT_EMBEDDING_DIM, optimization_strategy=None):
    embeddings = []
    total_sent_level_embs = 0
    first_debug_message = False

    for embedding_path in embeddings_path:
        # Get embedding (the returned embedding is not optimized)
        embedding = embedding_utils.load(embedding_path, dim=dim, strategy=optimization_strategy, to_float32=False)

        if not first_debug_message:
            first_debug_message = True

            logging.debug(f"First loaded embedding dtype: {embedding.dtype}")
            logging.debug(f"First loaded embedding shape: {embedding.shape}")

        embedding = embedding.astype(np.float32)

        total_sent_level_embs += embedding.shape[0]

        embeddings.append(embedding)

    logging.debug(f"Total sentence-level embeddings: {total_sent_level_embs}")

    return embeddings

def store_embeddings(output_path, embeddings, optimization_strategy=None):
    f = open(output_path, "wb")
    first_debug_message = False

    for embedding in embeddings:
        if optimization_strategy is not None:
            # Optimize embedding
            embedding = embedding_utils.get_optimized_embedding(embedding, strategy=optimization_strategy)

        if not first_debug_message:
            first_debug_message = True

            logging.debug(f"First stored embedding dtype: {embedding.dtype}")
            logging.debug(f"First stored embedding shape: {embedding.shape}")

        np.save(f, embedding)

    f.close()

def main(args):
    input_file = args.input_file
    output_file = args.output_file
    dim = args.dim
    optimization_strategy = None if args.optimization_strategy == 0 else args.optimization_strategy
    store_optimization_strategy = None if args.store_optimization_strategy == 0 else args.store_optimization_strategy

    if not os.path.isfile(input_file):
        raise FileNotFoundError(input_file)
    if os.path.isfile(output_file):
        raise exceptions.FileFoundError(output_file)

    if optimization_strategy is not None:
        logging.info(f"Loading embeddings with optimization strategy: {optimization_strategy}")
    if store_optimization_strategy is not None:
        logging.info(f"Storing embeddings with optimization strategy: {store_optimization_strategy}")

    embeddings_path = process_input_file(input_file)
    embeddings = load_embeddings(embeddings_path, dim=dim, optimization_strategy=optimization_strategy)

    logging.info(f"Loaded embeddings: {len(embeddings)}")

    store_embeddings(output_file, embeddings, optimization_strategy=store_optimization_strategy)

    logging.info(f"Embeddings have been stored: '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform individual sentence-embedding files to a merged embedding file')

    # Embedding
    parser.add_argument('input_file', metavar='input-file',
        help='TSV file with doc_path entries and without header')
    parser.add_argument('output_file', metavar='output-file',
        help='Output file where the embeddings will be stored')

    # Optional
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
