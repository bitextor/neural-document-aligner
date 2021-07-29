
import re
import os
import sys
import logging
import argparse
import tempfile

import numpy as np

import split_doc
import get_embedding
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

def generate_embeddings(batch_docs, batch_langs, batch_outputs, langs_to_process, max_size_per_batch, optimization_strategy=None,
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

        embedding_output = batch_outputs[idx]
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

    get_embedding.generate_and_store_embeddings(content, batch_outputs, no_sentences, optimization_strategy=optimization_strategy,
                                                input_is_list_of_sentences=True, model=model, batch_size=batch_size)

def buffered_read_from_list(inputs, buffer_size_mb):
    """Read from a list of inputs where each element is expected
       to be the path of a file which exists
    """
    buffer = []
    idx_start, idx_end = 0, 0
    current_bytes = 0

    for input in inputs:
        with open(input) as f:
            buffer.append(f.buffer.read())

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

def process_input_file(args, max_noentries=None):
    input_file = args.input_file
    force_generation = args.force_generation
    skip_embs_which_exist = args.skip_embs_which_exist

    docs = []
    langs = []
    embeddings_output = []
    file_open = False

    if input_file == "-":
        data_src = sys.stdin.readlines()
    else:
        data_src = open(input_file, "r", encoding="utf-8")
        file_open = True

    for line in data_src:
        for idx, line in enumerate(lines):
            if (max_noentries and idx >= max_noentries):
                logging.debug(f"The max. number of lines to process from the input file has been reached ({idx} lines processed from '{input_file}')")
                break

            line = line.strip().split("\t")

            docs.append(utils.expand_and_real_path_and_exists(line[0], raise_exception=True))
            langs.append(line[1])
            embeddings_output(utils.expand_and_real_path_and_exists(line[2], raise_exception=True, exception=FileFoundError,
                                                                    func_check_exists=lambda x: skip_embs_which_exist or force_generation or not os.path.isfile(x)))

            # Check if we need to remove the last element because the output already exists
            if skip_embs_which_exist:
                p = utils.expand_and_real_path_and_exists(line[2])

                if os.path.isfile(p):
                    # The output exists, so we need to remove the embedding generation of the current embedding

                    docs.pop()
                    langs.pop()
                    embeddings_output.pop()

    if file_open:
        data_src.close()

    return docs, langs, embeddings_output

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

    no_processed_files = 0
    no_processed_batches = 0
    noprocessed_files = 0
    noprocessed_batches = 0
    matched_batches_idx = 0
    size = 0

    # Process batches by size (max. size is 'max_size_per_batch')
    for batch, (batch_docs, bytes_length_batch, (idx_start, idx_end)) in enumerate(buffered_read_from_list(docs, max_size_per_batch)):
        size += bytes_length_batch

        logging.debug(f"Batch #{batch} of {float(bytes_length_batch / 1000.0 / 1000.0):.2f} MB from a max. of {max_size_per_batch} MB ({len(batch_docs)} lines)")

        if matched_batches_idx >= max_batches_process:
            logging.info("The max. number of batches to process have been reached")
            break

        # Check if it is our turn to process (group configuration)
        if matched_batches_idx % max_groups == group:
            logging.debug(f"Processing batch #{batch}")

            batch_langs = langs[idx_start:idx_end]
            batch_outputs = embeddings_output[idx_start:idx_end]

            logging.info(f"Size: {float(size) / 1000.0 / 1000.0:.2f} MB")
            logging.info(f"Langs which are going to be processed: {','.join(langs_to_process) if langs_to_process[0] != '-' else 'all languages'}")

            # Process the current batch
            generate_embeddings(batch_docs, batch_langs, batch_outputs, langs_to_process, max_size_per_batch,
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

    logging.info(f"Number of processed batches: {noprocessed_batches} of {noprocessed_batches + no_processed_batches}")
    logging.info(f"Number of processed files: {noprocessed_files} of {noprocessed_files + no_processed_files}")

def main(args):
    max_size_per_batch = args.max_mbytes_per_batch
    max_batches_process = args.max_batches_process
    group = args.group
    max_groups = args.max_groups
    langs_to_process = args.langs_to_process.split(",")
    optimization_strategy = args.optimization_strategy
    model = args.model
    batch_size = args.batch_size
    sentence_splitting = args.sentence_splitting

    docs, langs, embeddings_output = process_input_file(args)

    process(docs, langs, embeddings_output, langs_to_process=langs_to_process, max_mbytes_per_batch=max_size_per_batch,
            max_batches_process=max_batches_process, group=group, max_group=max_group, batch_size=batch_size,
            optimization_strategy=optimization_strategy, model=model, sentence_splitting=sentence_splitting)

def check_args(args):
    assert args.max_groups > 0, "The max. number of groups must be greater than 0"
    assert args.group >= 0, "The group ID must be greater or equal than 0"
    assert args.max_groups > args.group, "The group ID has to be fewer than the max. number of groups"
    assert args.langs_to_process != ""
    assert args.max_mbytes_per_batch > 0, "The max. MB has to be greater or equal than 0"
    assert not (args.force_generation and args.skip_those_which_exist), "The behaviour when generating embeddings in the case that the output already exists can be at the same time to overwrite and skip"

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
                        type=lambda x: is_file_arg(x, f=lambda x: (x == "-" or os.path.isfile(x))),
                        help='TSV file with doc_path\\tlang_of_doc\\tembedding_output_path entries and without header. If \'-\' is provided, instead of TSV file, values are expected to be provided via stdin')

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
    parser.add_argument('--force-generation', action='store_true',
                        help='Overwrite the output of embeddings instead of stopping the execution if the provided output exists')
    parser.add_argument('--skip-embs-which-exist', action='store_true',
                        help='Skip embeddings whose output already exist')
    parser.add_argument('--optimization-strategy', default=DEFAULT_VALUES['optimization_strategy'], type=int,
                        help='Store the embeddings using an optimization strategy. Default value is do not apply any optimization')
    parser.add_argument('--sentence-splitting', action='store_true',
                        help='Apply sentence splitting to the documents')
    # Logging
    parser.add_argument('--logging-level', metavar='N', type=int, default=DEFAULT_VALUES['logging_level'],
                        help=f'Logging level. Default value is {DEFAULT_VALUES["logging_level"]}')

    args = parser.parse_args()

    utils.set_up_logging(level=args.logging_level)

    check_args(args)

    main(args)
