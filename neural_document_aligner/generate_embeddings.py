#!/usr/bin/env python3

import re
import os
import sys
import base64
import logging
import argparse
import tempfile

import numpy as np
from sentence_transformers import SentenceTransformer, util

import utils.embedding_utils as embedding_utils
import split_doc
import utils.utils as utils
from exceptions import FileFoundError
import utils.utils as utils
import constants

DEFAULT_VALUES = {
    "langs_to_process": "-",
    "max_mbytes_per_batch": constants.DEFAULT_MAX_MBYTES_PER_BATCH,
    "max_batches_process": np.inf,
    "group": 0,
    "max_groups": 1,
    "logging_level": 20,
    "optimization_strategy": None,
    "model": constants.DEFAULT_ST_MODEL,
    "batch_size": constants.DEFAULT_BATCH_SIZE,
    "sentence_splitting": constants.DEFAULT_SENTENCE_SPLITTING,
}

def get_embedding_from_sentence_transformer(list_of_sentences, optimization_strategy=None, model=constants.DEFAULT_EMBEDDING_DIM,
                                            batch_size=constants.DEFAULT_BATCH_SIZE):
    try:
        model = SentenceTransformer(model)
    except Exception as e:
        raise Exception(f"could not load the model '{model}' (maybe is not in the list of available models of 'sentence_transformers')") from e

    embeddings = model.encode(list_of_sentences, batch_size=batch_size, show_progress_bar=constants.ST_SHOW_PROGRESS)

    # Embedding optimization
    if optimization_strategy is not None:
        embeddings = embedding_utils.get_optimized_embedding(embeddings, strategy=optimization_strategy)

    return embeddings

def generate_and_store_embeddings(input, output_fd, no_sentences, optimization_strategy=None, input_is_list_of_sentences=False,
                                  model=constants.DEFAULT_EMBEDDING_DIM, batch_size=constants.DEFAULT_BATCH_SIZE):
    list_of_sentences = input

    if not input_is_list_of_sentences:
        list_of_sentences = []

        with open(input, "r") as f:
            for line in f:
                list_of_sentences.append(line.strip())

    embeddings = get_embedding_from_sentence_transformer(list_of_sentences, optimization_strategy=optimization_strategy, model=model,
                                                         batch_size=batch_size)

    previous = 0

    for no_sent in no_sentences:
        emb = embeddings[previous:previous + no_sent]
        previous += no_sent

        np.save(output_fd, emb)

    if embeddings.shape[0] != previous:
        logging.warning("Once the embedding has been stored, the length seems to mismatch")

def generate_embeddings(batch_docs, batch_langs, embeddings_output_fd, langs_to_process, max_size_per_batch, optimization_strategy=None,
                        model=None, batch_size=DEFAULT_VALUES["batch_size"], sentence_splitting=DEFAULT_VALUES["sentence_splitting"]):
    if model is None:
        model = DEFAULT_VALUES["model"]

    content = []
    no_sentences = []
    output_files = []
    total_no_sentences = 0

    for idx, doc_content in enumerate(batch_docs):
        current_lang = batch_langs[idx]

        if ("-" not in langs_to_process and current_lang not in langs_to_process):
            continue

        document_content = doc_content

        if sentence_splitting:
            split_status, document_content_splitted = split_doc.split(None, current_lang, output=None, text=document_content)
#            document_content_splitted = document_content_splitted.strip().split("\n")

            if split_status != 0:
                logging.warning(f"Splitting status is not 0: {split_status}")
        else:
            document_content_splitted = document_content

        document_content_splitted = document_content_splitted.strip().split("\n")

        content.extend(document_content_splitted)
        no_sentences.append(len(document_content_splitted))
        total_no_sentences += no_sentences[-1]

    if len(content) != total_no_sentences:
        logging.warning(f"The number of sentences do not match with the expected ({len(content)} vs {total_no_sentences})")

    generate_and_store_embeddings(content, embeddings_output_fd, no_sentences, optimization_strategy=optimization_strategy,
                                  input_is_list_of_sentences=True, model=model, batch_size=batch_size)

def buffered_read_from_list(inputs, buffer_size_mb, docs_are_base64_values=False):
    """Read from a list of inputs where each element is expected
       to be the path of a file which exists
    """
    buffer = []
    idx_start, idx_end = 0, 0
    current_bytes = 0

    for input in inputs:
        # Append file content (binary)
        if docs_are_base64_values:
            buffer.append(base64.b64decode(input))
        else:
            with open(input) as f:
                buffer.append(f.buffer.read())

        # Process binary data

        current_bytes += len(buffer[-1])

        buffer[-1] = buffer[-1].decode("utf-8").strip()

        if current_bytes >= buffer_size_mb * 1000 * 1000:
            idx_end += len(buffer)

            yield buffer, current_bytes, (idx_start, idx_end)

            buffer = []
            current_bytes = 0
            idx_start = idx_end

    if len(buffer) > 0:
        yield buffer, current_bytes, (idx_start, idx_end + len(buffer))

def process_input_file(input_file, max_noentries=None):
    docs = []
    langs = []
    file_open = False
    format_lines = None

    if input_file == "-":
        data_src = sys.stdin.readlines()
    else:
        data_src = open(input_file, "r", encoding="utf-8")
        file_open = True

    for idx, line in enumerate(data_src):
        if (max_noentries and idx >= max_noentries):
            logging.debug(f"The max. number of lines to process from the input file has been reached ({idx} lines processed from '{input_file}')")
            break

        line = line.strip().split("\t")

        if format_lines is None:
            format_lines = len(line)

            if format_lines not in (1, 2):
                raise Exception("detected format is not correct")

            logging.debug(f"Using format with {format_lines} lines (input file)")

        if len(line) == format_lines:
            docs.append(utils.expand_and_real_path_and_exists(line[0], raise_exception=True))

            if format_lines == 2:
                langs.append(line[1])

        else:
            logging.warning("Unexpected format in line #{idx + 1} (it will be skipped)")
            continue

    if (len(langs) == 0 or len(langs) != len(docs)):
        logging.debug("Langs were not provided")

        langs = [None] * len(docs)

    if file_open:
        data_src.close()

    return docs, langs

def process(docs, langs, embeddings_output, **kwargs):
    max_size_per_batch = kwargs["max_mbytes_per_batch"] if "max_mbytes_per_batch" in kwargs else DEFAULT_VALUES["max_mbytes_per_batch"]
    max_batches_process = kwargs["max_batches_process"] if "max_batches_process" in kwargs else DEFAULT_VALUES["max_batches_process"]
    group = kwargs["group"] if "group" in kwargs else DEFAULT_VALUES["group"]
    max_groups = kwargs["max_groups"] if "max_groups" in kwargs else DEFAULT_VALUES["max_groups"]
    langs_to_process = kwargs["langs_to_process"] if "langs_to_process" in kwargs else DEFAULT_VALUES["langs_to_process"]
    optimization_strategy = kwargs["optimization_strategy"] if "optimization_strategy" in kwargs else DEFAULT_VALUES["optimization_strategy"]
    model = kwargs["model"] if "model" in kwargs else DEFAULT_VALUES["model"]
    batch_size = kwargs["batch_size"] if "batch_size" in kwargs else DEFAULT_VALUES["batch_size"]
    sentence_splitting = kwargs["sentence_splitting"] if "sentence_splitting" in kwargs else DEFAULT_VALUES["sentence_splitting"]
    docs_are_base64_values = kwargs["docs_are_base64_values"] if "docs_are_base64_values" in kwargs else False

    no_processed_files = 0
    no_processed_batches = 0
    noprocessed_files = 0
    noprocessed_batches = 0
    matched_batches_idx = 0
    size = 0
    embeddings_output_fd = open(embeddings_output, "wb")

    if len(docs) != len(langs):
        raise Exception("the length of the provided docs and langs do not match")
    if len(docs) == 0:
        raise Exception("the length of the provided docs is 0")
    if (sentence_splitting and langs[0] is None):
        raise Exception(f"you want to sentence-splitting but did not provide the langs, which is necessary")

    # Process batches by size (max. size is 'max_size_per_batch')
    for batch, (batch_docs, bytes_length_batch, (idx_start, idx_end)) in enumerate(buffered_read_from_list(docs, max_size_per_batch, docs_are_base64_values=docs_are_base64_values)):
        size += bytes_length_batch

        logging.debug(f"Batch #{batch} of {float(bytes_length_batch / 1000.0 / 1000.0):.2f} MB from a max. of {max_size_per_batch} MB ({len(batch_docs)} lines)")

        if matched_batches_idx >= max_batches_process:
            logging.info("The max. number of batches to process have been reached")
            break

        # Check if it is our turn to process (group configuration)
        if matched_batches_idx % max_groups == group:
            logging.debug(f"Processing batch #{batch}")

            batch_langs = langs[idx_start:idx_end]

            logging.info(f"Size: {float(size) / 1000.0 / 1000.0:.2f} MB")
            logging.info(f"Langs which are going to be processed: {','.join(langs_to_process) if langs_to_process[0] != '-' else 'all languages'}")

            # Process the current batch
            generate_embeddings(batch_docs, batch_langs, embeddings_output_fd, langs_to_process, max_size_per_batch,
                                optimization_strategy=optimization_strategy, model=model, batch_size=batch_size)

            size = 0
            noprocessed_batches += 1
            noprocessed_files += len(batch_docs)
        else:
            # It was not our turn to process. This batch corresponds to other group to process it

            logging.debug(f"Batch #{batch} was not processed because is the job of other group (we are the group {group})")

            no_processed_batches += 1
            no_processed_files += len(batch_docs)

        matched_batches_idx += 1

    embeddings_output_fd.close()

    logging.info(f"Number of processed batches: {noprocessed_batches} of {noprocessed_batches + no_processed_batches}")
    logging.info(f"Number of processed files: {noprocessed_files} of {noprocessed_files + no_processed_files}")

def main(args):
    input_file = args.input_file
    output_file = args.output_file
    max_size_per_batch = args.max_mbytes_per_batch
    max_batches_process = args.max_batches_process
    group = args.group
    max_groups = args.max_groups
    langs_to_process = args.langs_to_process.split(",")
    optimization_strategy = args.optimization_strategy
    model = args.model
    batch_size = args.batch_size
    sentence_splitting = args.sentence_splitting
    paths_to_docs_are_base64_values = args.paths_to_docs_are_base64_values

    docs, langs = process_input_file(input_file)

    process(docs, langs, output_file, langs_to_process=langs_to_process, max_mbytes_per_batch=max_size_per_batch,
            max_batches_process=max_batches_process, group=group, max_group=max_group, batch_size=batch_size,
            optimization_strategy=optimization_strategy, model=model, sentence_splitting=sentence_splitting,
            docs_are_base64_values=paths_to_docs_are_base64_values)

def check_args(args):
    errors = [
        (args.max_groups <= 0, "The max. number of groups must be greater than 0"),
        (args.group < 0, "The group ID must be greater or equal than 0"),
        (args.max_groups <= args.group, "The group ID has to be fewer than the max. number of groups"),
        (args.langs_to_process == "", "The provided langs to process cannot contain an empty string"),
        (args.max_mbytes_per_batch <= 0, "The max. MB has to be greater or equal than 0"),
    ]

    for condition, msg in errors:
        if condition:
            raise Exception(msg)

def is_file(file, f=os.path.isfile):
    if not f(os.path.abspath(os.path.expanduser(file))):
        return False

    return True

def is_file_arg(file, f=os.path.isfile):
    if not is_file(file, f=f):
        msg = "Could not find the provided path"

        raise argparse.ArgumentTypeError(msg)
    return file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate embeddings with TSV files')

    parser.add_argument('input_file', metavar='input-file',
                        type=lambda x: is_file_arg(x, f=lambda y: (y == "-" or os.path.isfile(x))),
                        help='TSV file with doc_path<tab>lang_of_doc entries and without header, but the langs are optional. If \'-\' is provided, instead of TSV file, values are expected to be provided via stdin')
    parser.add_argument('output_file', metavar='output-file',
                        type=lambda x: not is_file_arg(x),
                        help='Path were the embeddings will be stored. In order to recover them, you will need to use np.load() as many times as necessary to reach the aimed document')

    parser.add_argument('--group', type=int, default=DEFAULT_VALUES['group'], metavar='N',
                        help='Group ID which the execution will use in order to only process those files which, sequentially, must process. This flag is used to make '
                             'the paralization easier across different processes executions. The distribution of work is linear. The default value is to be the group 0')
    parser.add_argument('--max-groups', type=int, default=DEFAULT_VALUES['max_groups'], metavar='N',
                        help='Max. number of groups which are executing across different processes. Check the \'--group\' flag for more information. The default value is 1')
    parser.add_argument('--max-batches-process', type=int, default=DEFAULT_VALUES['max_batches_process'], metavar='N',
                        help='Max. number of batches to process. You should take into account that even the non-processed batches due to groups configuration will be counted. The default value is no limit')
    parser.add_argument('--max-mbytes-per-batch', type=int, default=DEFAULT_VALUES['max_mbytes_per_batch'], metavar='N',
                        help='Max. MB which will be processed in a batch. The provided size is not guaranteed, since once it has been detected that the batch contains that size, it will be processed. The default value is {DEFAULT_VALUES["max_mbytes_per_batch"]} (MB)')
    parser.add_argument('--model', default=DEFAULT_VALUES['model'], metavar='MODEL',
                        help=f'Model to use from \'sentence_transformers\'. The default model is \'{DEFAULT_VALUES["model"]}\'')
    parser.add_argument('--batch-size', default=DEFAULT_VALUES['batch_size'], metavar='N',
                        help=f'Batch size for the embeddings generation. The default value is \'{DEFAULT_VALUES["batch_size"]}\'')

    # Lang
    parser.add_argument('--langs-to-process', default=DEFAULT_VALUES['langs_to_process'],
                        help='Langs which are going to be processed. The provided langs have to be separatted by \',\'. The default value is process all of them')

    # Other
    parser.add_argument('--optimization-strategy', default=DEFAULT_VALUES['optimization_strategy'], type=int,
                        help='Store the embeddings using an optimization strategy. Default value is do not apply any optimization')
    parser.add_argument('--sentence-splitting', action='store_true',
                        help='Apply sentence splitting to the documents. You will need to provide the langs in the input file if you want to apply the sentence splitter')
    parser.add_argument('--paths-to-docs-are-base64-values', action='store_true',
                        help='The first column of the input file is expected to be paths to docs. If this option is set, the expected value will be the base64 value of the docs instead')
    # Logging
    parser.add_argument('--logging-level', metavar='N', type=int, default=DEFAULT_VALUES['logging_level'],
                        help=f'Logging level. Default value is {DEFAULT_VALUES["logging_level"]}')

    args = parser.parse_args()

    utils.set_up_logging(level=args.logging_level)

    check_args(args)

    main(args)
