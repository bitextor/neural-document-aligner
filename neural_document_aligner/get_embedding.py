#!/usr/bin/python3

import os
import sys
import logging
import argparse
import tempfile
import subprocess
import numpy as np

from sentence_transformers import SentenceTransformer, util

import utils.embedding_util as embedding_util
import constants

def get_embedding(embedding_file, dim=constants.DEFAULT_EMBEDDING_DIM, optimization_strategy=None):
    embedding = embedding_util.load(embedding_file, dim=dim, strategy=optimization_strategy)

    return embedding

def get_embedding_from_sentence_transformer(list_of_sentences, optimization_strategy=None, model=constants.DEFAULT_EMBEDDING_DIM):
    try:
        model = SentenceTransformer(model)
    except Exception as e:
        raise Exception(f"could not load the model '{model}' (maybe is not in the list of available models of 'sentence_transformers')") from e

    embeddings = model.encode(list_of_sentences)

    # Embedding optimization
    if optimization_strategy is not None:
        embeddings = embedding_util.get_optimized_embedding(embeddings, strategy=optimization_strategy)

    return embeddings

def generate_and_store_embeddings(input, outputs, no_sentences, optimization_strategy=None, input_is_list_of_sentences=False,
                                  model=constants.DEFAULT_EMBEDDING_DIM):
    list_of_sentences = input

    if not input_is_list_of_sentences:
        list_of_sentences = []

        with open(input, "r") as f:
            for line in f:
                list_of_sentences.append(line.strip())

    embeddings = get_embedding_from_sentence_transformer(list_of_sentences, optimization_strategy=optimization_strategy, model=model)

    previous = 0

    for no_sent, output in zip(no_sentences, outputs):
        emb = embeddings[previous:previous + no_sent]
        previous += no_sent

        emb.tofile(output)

    if embeddings.shape[0] != previous:
        logging.warning("Once the embedding has been stored, the length seems to mismatch")
