#!/usr/bin/python3

import os
import sys
import logging
import argparse
import tempfile
import subprocess
import numpy as np

import utils.embedding_util as embedding_util

DEFAULT_EMBEDDING_DIM = 1024

# get LASER path
if os.environ.get("LASER") is None:
    raise Exception("environment variable LASER not set")

LASER = os.environ['LASER']

def get_embedding(embedding_file, dim=DEFAULT_EMBEDDING_DIM, optimization_strategy=None):
    embedding = embedding_util.load(embedding_file, dim=dim, strategy=optimization_strategy)

    return embedding

def get_embedding_from_laser(input_file, lang, output_fifo_filename="laser_embeddings", optimization_strategy=None,
                             dim=DEFAULT_EMBEDDING_DIM, verbose=False, content=None, embed_script_path=None):
    if ((input_file is None and content is None) or
        (input_file is not None and content is not None)):
        raise Exception("you have to provide the data either with an input file or with the content itself")

    file_desc = None

    if input_file == "-":
        cat = subprocess.Popen(["cat"], stdin=sys.stdin, stdout=subprocess.PIPE)
    elif input_file is not None:
        file_desc = open(input_file, "r")
        cat = subprocess.Popen(["cat"], stdin=file_desc, stdout=subprocess.PIPE)

    laser_encoder = f"{LASER}/models/bilstm.93langs.2018-12-26.pt"
    laser_bpe_codes = f"{LASER}/models/93langs.fcodes"

    tmpdir = tempfile.mkdtemp()
    fifo_filename = os.path.join(tmpdir, output_fifo_filename)

    try:
        os.mkfifo(fifo_filename)
    except OSError as e:
        logging.error(str(e))
        logging.error(f"could not create the FIFO '{output_fifo_filename}' (returning all zeros with size {dim})")

        emb = np.zeros(dim, dtype=np.float32)

        if optimization_strategy is not None:
            return embedding_util.get_optimized_embedding(emb, optimization_strategy)

        return emb

    logging.info(f"using FIFO '{fifo_filename}'")

    embedding = b""
    embed_script_path = f"{LASER}/source/embed.py" if embed_script_path is None else embed_script_path

    popen_command = ["python3", embed_script_path,
                    "--encoder", laser_encoder,
                    "--token-lang", lang,
                    "--bpe-codes", laser_bpe_codes,
                    "--output", fifo_filename,
                    "--np-savetxt"]

    laser = subprocess.Popen(popen_command, stdin=cat.stdout)

    fifo = open(fifo_filename, "r") # Waits until FIFO is opened to write
    value = True

    # Read until the process finishes
    while laser.poll() is None:
        value = fifo.buffer.read()

        if not value:
            continue

        embedding += value

    fifo.close()
    os.remove(fifo_filename)
    os.rmdir(tmpdir)

    if file_desc is not None:
        file_desc.close()

    embedding = np.fromstring(embedding, dtype=np.float32, sep="\n")
    embedding.resize(embedding.shape[0] // dim, dim)

    # Embedding optimization
    if optimization_strategy is not None:
        embedding = embedding_util.get_optimized_embedding(embedding, optimization_strategy)

    return embedding

def generate_and_store_embeddings(input, output, lang, no_sentences, optimization_strategy=None, embed_script_path=None):
    total_no_sentences = np.sum(no_sentences)

    embedding = get_embedding_from_laser(input, lang, optimization_strategy=optimization_strategy, embed_script_path=embed_script_path)

    if embedding.shape[0] != total_no_sentences:
        logging.warning(f"the resulting embedding length ({embedding.shape[0]}) mismatches with the number of sentences ({total_no_sentences}). Writting all embeddings in just one file: '{output[0]}.merged'")

        embedding.tofile(f"{output[0]}.merged")
    else:
        previous = 0

        for no_s, o in zip(no_sentences, output):
            e = embedding[previous:previous + no_s]
            previous += no_s

            e.tofile(o)

        if embedding.shape[0] != previous:
            logging.warning("once the embedding has been stored, the length seems to mismatch")
