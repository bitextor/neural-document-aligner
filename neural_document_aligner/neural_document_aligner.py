#!/usr/bin/python3

import os
import sys
import math
import copy
import logging
import argparse
import subprocess
import numpy as np
from scipy import spatial
from operator import itemgetter
from multiprocessing import Process, Pool

import faiss

import utils.utils as utils
import get_embedding
import generate_embeddings as gen_embeddings
import evaluate
import levenshtein
from exceptions import FileFoundError

DEFAULT_EMBEDDING_DIM = 1024

def generate_embeddings(docs, embedding_files, lang, they_should_exist=True, optimization_strategy=None, embed_script_path=None):
    for idx, embedding in enumerate(embedding_files):
        # Check if the embeddings either should or not exist
        if os.path.isfile(embedding) != they_should_exist:
            if they_should_exist:
                logging.error(f"embedding #{idx} should exist but it does not")
                raise FileNotFoundError(embedding)
            else:
                logging.error(f"embedding #{idx} should not exist but it does")
                raise FileFoundError(embedding)

    if not they_should_exist:
        # Generate embeddings because they should not exist
        gen_embeddings.process(docs, [lang] * len(docs), embedding_files, optimization_strategy=optimization_strategy, embed_script_path=embed_script_path)

def get_embedding_vectors(embedding_file, dim=DEFAULT_EMBEDDING_DIM, optimization_strategy=None):
    embedding = get_embedding.get_embedding(embedding_file, dim=dim, optimization_strategy=optimization_strategy)

    return embedding

def cosine_similarity(v1, v2, clipping=True):
    cosine_similarity = spatial.distance.cosine(v1, v2) # Where 0 means vectors with same direction and sense and 1 orthogonal and 2 opposite
    cosine_distance = 1.0 - cosine_similarity # Where -1 means opposite, 0 orthogonal and 1 vectors with same direction and sense

    # Clipping
    if (clipping and cosine_distance < 0.0):
        cosine_distance = 0.0
        cosine_similarity = 1.0 - cosine_distance # Where 0 means equal and 1 different. [0,1] because of clipping

    return cosine_similarity

def median_embedding(embedding):
    return np.median(embedding, axis=0)

def max_embedding(embedding):
    return np.amax(embedding, axis=0)

def get_weights_sl(path, cnt=len):
    file = open(path, "r")
#    lines = subprocess.check_output(f"sort {path}", shell=True).decode("utf-8").strip().split("\n")
#    nolines = 0
#    weights = np.zeros(len(lines), dtype=np.float32)
    weights = []
    counts = {}
    lengths = {}
    lengths_sum = 0.0

    # Store the length and the times a sentence appears
    for l in file:
#    for l in lines:
        l = l.strip()
        h = hash(l)

        if h not in counts:
            counts[h] = 0
            lengths[h] = cnt(l)

        counts[h] += 1

    file.close()

    # Iterate over found sentences and calculate summ(times * length)
    for h in counts:
        lengths_sum += counts[h] * lengths[h]

    if lengths_sum == 0:
        logging.warning("empty or near-empty file -> not applying weights (i.e. weights = [1.0, ..., 1.0])")

    file = open(path, "r")

    for l in file:
        l = l.strip()
        h = hash(l)

        if lengths_sum == 0:
            weights.append(1.0)
        else:
#            weights[nolines] = (counts[h] * lengths[h]) / lengths_sum
            weights.append((counts[h] * lengths[h]) / lengths_sum)
#            nolines += 1

    file.close()

    return np.float32(weights)

def get_weights_sl_all(paths, cnt=len):
    results = []

    for path in paths:
        weights = get_weights_sl(path, cnt=cnt)

        results.append(weights)

    return results

# paths arg must contain src and trg docs
def get_weights_idf(paths, path):
    nodocs = len(paths)
    weights = []
    idx = {}
    file = open(path, "r")

    # Initializate weights and store indexes of sentences by the hash of the sentence
    for i, l in enumerate(file):
        l = l.strip()
        h = hash(l)

        if h not in idx:
            idx[h] = set()

        idx[h].add(i)

        weights.append(0)
    file.close()

    # Iterate over all the docs and check if any of them contains any of the previous indexed sentences
    # If contains: increment weights in 1 (it indicates the quantity of times a sentence is found, at least once, in a doc) and do not increment again for the same sentence in the same doc
    for p in paths:
        loop_file = open(p, "r")
        found_sentences = set()

        for l in loop_file:
            l = l.strip()
            h = hash(l)

            if (h in idx and h not in found_sentences):
                for widx in idx[h]:
                    weights[widx] += 1

                found_sentences.add(h)

        loop_file.close()

    for i, w in enumerate(weights):
        weights[i] = 1.0 + np.log(nodocs / w)

    return np.float32(weights)

# paths arg must contain src and trg docs
def get_weights_idf_all(paths):
    results = []
    nodocs = len(paths)
    idx = {} # index by hash(sentence) which contains the number of times the sentence is in a document (at least once)

    # Iterate over all docs and count iteratively the times a sentences is in a document at least once
    for p in paths:
        loop_file = open(p, "r")
        found_sentences = set()

        for l in loop_file:
            l = l.strip()
            h = hash(l)

            if h not in idx:
                # Once per sentence
                idx[h] = 0

            if h not in found_sentences:
                # Once per sentence per document
                idx[h] += 1
                found_sentences.add(h)

        loop_file.close()

    # Iterate over all docs and calculate idf for every doc
    for p in paths:
        loop_file = open(p, "r")
        weights = []

        for l in loop_file:
            l = l.strip()
            h = hash(l)
            r = 1.0 + np.log(nodocs / idx[h])

            weights.append(r)

        results.append(np.float32(weights))

        loop_file.close()

    return results

def get_weights_slidf(paths, path, cnt=len):
    sl_weights = get_weights_sl(path, cnt)
    idf_weights = get_weights_idf(paths, path)

    if len(idf_weights) != len(sl_weights):
        logging.warning(f"different length of idf weights ({len(idf_weights)}) and sl weights ({len(sl_weights)})")

    return sl_weights * idf_weights

def get_weights_slidf_all(paths, cnt=len):
    results = []

    idf_weights = get_weights_idf_all(paths)
    sl_weights = []

    for p in paths:
        sl_weights.append(get_weights_sl(p, cnt))

    if len(idf_weights) != len(sl_weights):
        logging.warning(f"different length of idf weights ({len(idf_weights)}) and sl weights ({len(sl_weights)})")

    for idx in range(len(sl_weights)):
        if len(sl_weights[idx]) != len(idf_weights[idx]):
            logging.warning(f"different length of idf weights ({len(idf_weights)}) and sl weights ({len(sl_weights)}) for index {idx}, whose document should be '{paths[idx]}'")

        results.append(sl_weights[idx] * idf_weights[idx])

    return results
#    return get_weights_sl(path, cnt) * get_weights_idf(paths, path)

def weight_embeddings(embeddings, paths, weights_strategy=0):
    if weights_strategy == 0:
        # Do not apply any strategy
        return embeddings

#    if len(embs.shape) != 3:
#        raise Exception(f"unexpected embeddings shape before apply weights ({len(embs.shape)} vs 3)")
    if len(paths) != len(embeddings):
        raise Exception(f"the number of embeddings do not match with the number of paths ({len(embeddings)} vs {len(paths)})")

    get_weights_function = None
    weights_args = None
    weights_kwargs = None

    if weights_strategy == 1:
        get_weights_function = get_weights_sl_all
        weights_args = [paths]
        weights_kwargs = {}
    elif weights_strategy == 2:
        get_weights_function = get_weights_idf_all
        weights_args = [paths]
        weights_kwargs = {}
    elif weights_strategy == 3:
        get_weights_function = get_weights_slidf_all
        weights_args = [paths]
        weights_kwargs = {}
    else:
        raise Exception(f"unknown weight strategy: {weights_strategy}")

    weights = get_weights_function(*weights_args, **weights_kwargs)
#    weights = np.array(weights)

    if len(embeddings) != len(weights):
        logging.warning(f"shapes between embeddings and weights do not match ({len(embeddings)} vs {len(weights)}), so no weights are going to be applied")
    else:
        # Apply weights
        for emb_idx, (embedding, weight) in enumerate(zip(embeddings, weights)):
            embs = np.array(embedding, dtype=np.float32)
            weight = np.array(weight)

            if embs.shape[0] != weight.shape[0]:
                logging.warning(f"shapes between embedding and weights do not match ({embs.shape[0]} vs {weight.shape[0]}), so no weights are going to be applied to the current embedding (idx {emb_idx})")
                continue

            for idx in range(len(embs)):
                embs[idx] *= weight[idx]

            embeddings[emb_idx] = embs

    return embeddings

def merge_embedding(doc_embeddings, merging_strategy=0, dim=DEFAULT_EMBEDDING_DIM):
    result = None

    if merging_strategy == 0:
        return doc_embeddings

    if merging_strategy == 1:
        result = average_embedding(doc_embeddings)
    elif merging_strategy == 2:
        result = median_embedding(doc_embeddings)
    elif merging_strategy == 3:
        result = max_embedding(doc_embeddings)
    elif merging_strategy == 4:
        result = max_split3_embedding(doc_embeddings, dim=dim)
    elif merging_strategy == 5:
        result = iterative_average_embedding(doc_embeddings)
    else:
        raise Exception(f"unknown merging strategy: {merging_strategy}")

    return result

def max_split3_embedding(embedding, dim=DEFAULT_EMBEDDING_DIM):
    result = []
    idxs = [len(embedding)]

    if len(embedding) >= 3:
        first_idx = int(len(embedding) / 3)
        second_idx = first_idx + len(embedding) - 2 * first_idx

        idxs.insert(-1, first_idx)
        idxs.insert(-1, second_idx)
    elif len(embedding) == 2:
        result = np.concatenate([result, embedding[0]])
        result = np.concatenate([result, embedding[1]])
        result = np.concatenate([result, max_embedding(embedding)])
    else:
        result = np.concatenate([result, embedding[0]])
        result = np.concatenate([result, embedding[0]])
        result = np.concatenate([result, embedding[0]])

    if len(result) == 0:
        last_idx = 0

        for idx in idxs:
            max_result = max_embedding(embedding[last_idx:idx])

            result = np.concatenate([result, max_result])

            last_idx = idx

    if len(result) != dim * 3:
        logging.warning(f"unexpected size of result in max_split3_embedding ({len(result)})")

    return np.float32(result)

def iterative_average_embedding(embedding):
    result = embedding[0]

    for i in range(len(embedding) - 1):
        result = np.mean([result, embedding[i+1]], axis=0)

    return result

def average_embedding(embedding):
    assert len(embedding.shape) == 2, f"Unexpected shape ({len(embedding.shape)} vs 2)"

    return np.mean(embedding, axis=0, dtype=embedding.dtype)

def average_similarity(embedding_src, embedding_trg, storage=None):
    avg_embedding_src_vector = None
    avg_embedding_trg_vector = None
    src_key = hash(str(embedding_src))
    trg_key = hash(str(embedding_trg))

    if isinstance(storage, dict):
        if src_key in storage:
            avg_embedding_src_vector = storage[src_key]
        if trg_key in storage:
            avg_embedding_trg_vector = storage[trg_key]

    if avg_embedding_src_vector is None:
        avg_embedding_src_vector = average_embedding(embedding_src)

        if isinstance(storage, dict):
            storage[src_key] = avg_embedding_src_vector

    if avg_embedding_trg_vector is None:
        avg_embedding_trg_vector = average_embedding(embedding_trg)

        if isinstance(storage, dict):
            storage[trg_key] = avg_embedding_trg_vector

    cosine = cosine_similarity(avg_embedding_src_vector, avg_embedding_trg_vector)

    return 1.0 - cosine

def levenshtein_norm_factor(src_embeddings, trg_embeddings):
    levenshtein_nfactor = 0

    for src_embedding in src_embeddings:
        levenshtein_nfactor = max(levenshtein_nfactor, len(src_embedding))

    for trg_embedding in trg_embeddings:
        levenshtein_nfactor = max(levenshtein_nfactor, len(trg_embedding))

    return levenshtein_nfactor

def worker_lev(embedding_src, embedding_trg, levenshtein_nfactor, full, return_value):
    if full:
        lev_result = levenshtein.levenshtein(embedding_src, embedding_trg, nfactor=levenshtein_nfactor,
                                             diff_function_bool=lambda x, y: True,
                                             diff_function_value=lambda x, y: cosine_similarity(x, y))
    else:
        lev_result = levenshtein.levenshtein_opt_space_and_band(embedding_src, embedding_trg, nfactor=levenshtein_nfactor,
                                                                diff_function_bool=lambda x, y: True,
                                                                diff_function_value=lambda x, y: cosine_similarity(x, y))

    return (lev_result, return_value)

def worker_avg(embedding_src, embedding_trg, return_value):
    avg_result = average_similarity(embedding_src, embedding_trg)

    return (avg_result, return_value)

def worker_distance(embedding_src, embedding_trg, return_value):
    cosine = cosine_similarity(avg_embedding_src_vector, avg_embedding_trg_vector)

    return (1.0 - cosine, return_value)

def docalign(results, src_docs, trg_docs, src_urls, trg_urls, output_with_urls, only_docalign=False):
    result = {'src': {}, 'trg': {}}

    if (len(results) == 0 or len(results[0]) == 0):
        return result

    if only_docalign:
        result = set()

        for r in results:
            src_doc = r[0]
            trg_doc = r[1]

            if output_with_urls:
                src_idx = src_docs.index(src_doc)
                src_doc = src_urls[src_idx]

                trg_idx = trg_docs.index(trg_doc)
                trg_doc = trg_urls[trg_idx]

            result.add((src_doc, trg_doc))

        return result

    for r in results:
        src_doc = r[0]
        trg_doc = r[1]

        if output_with_urls:
            src_idx = src_docs.index(src_doc)
            src_doc = src_urls[src_idx]

            trg_idx = trg_docs.index(trg_doc)
            trg_doc = trg_urls[trg_idx]

        score = r[-1]

        if src_doc not in result['src'].keys():
            result['src'][src_doc] = None

        if trg_doc not in result['trg'].keys():
            result['trg'][trg_doc] = None

            # Check if current score is higher from source to target
        if (result['src'][src_doc] is None or result['src'][src_doc][-1] < score):
            result['src'][src_doc] = [trg_doc, score]

            # Check if current score is higher from target to source
        if (result['trg'][trg_doc] is None or result['trg'][trg_doc][-1] < score):
            result['trg'][trg_doc] = [src_doc, score]

    return result

def union_and_intersection(aligned_urls):
    union = set()
    intersection = set()

    # Iterate from source to target
    for src_au in aligned_urls['src']:
        src_url = src_au
        trg_url = aligned_urls['src'][src_url][0]

        best_trg_url = aligned_urls['trg'][trg_url][0]

        if src_url == best_trg_url:
            intersection.add((src_url, trg_url))

        union.add((src_url, trg_url))

    # Iterate from target to source
    for trg_au in aligned_urls['trg']:
        trg_url = trg_au
        src_url = aligned_urls['trg'][trg_url][0]

        union.add((src_url, trg_url))

    return {'union': union, 'intersection': intersection}

def get_docs(path, max_nodocs=None, iso88591=False):
    docs = []
    urls = []
    idx = 0

    if iso88591:
        f = open(path, "r", encoding="ISO-8859-1")
    else:
        f = open(path, "r")

    try:
        for line in f:
            if (max_nodocs and idx >= max_nodocs):
                break

            line = line.strip().split("\t")

            docs.append(line[0][:-7])
            urls.append(line[1])

            idx += 1
    except Exception as e:
        logging.error(str(e))

        if not iso88591:
            del docs
            del urls

            logging.warning(f"trying with ISO-8859-1 encoding")

            return get_docs(path, max_nodocs, True)

    f.close()

    return docs, urls

def process_input_file(args, max_noentries=None):
    input_file = args.input_file
    input_src_and_trg_splitted = args.input_src_and_trg_splitted
    generate_embeddings_arg = args.generate_embeddings

    src_docs, trg_docs = [], []
    src_embedding_files, trg_embedding_files = [], []
    src_urls, trg_urls = [], []
    get_urls = True
    file_open = False

    if input_file == "-":
        data_src = sys.stdin.readlines()
    else:
        data_src = open(input_file, "r", encoding="utf-8")
        file_open = True

    for idx, line in enumerate(data_src):
#        for idx, line in enumerate(lines):
        if (max_noentries and idx >= max_noentries):
            logging.debug(f"max. number of lines to process from the input file has been reached ({idx} lines processed from '{input_file}')")
            break

        line = line.strip().split("\t")

        if input_src_and_trg_splitted:
            # Expected format: doc\tembedding_file\turl_or_dash\t<src|trg>

            if len(line) != 4:
                logging.warning(f"unexpected format in line #{idx + 1} (it will be skipped)")
                continue

            src_or_trg = line[3].lower()

            if src_or_trg not in ("src", "trg"):
                logging.warning(f"unexpected format in line #{idx + 1} (it will be skipped)")
                continue

            docs_vector = src_docs
            embedding_files_vector = src_embedding_files
            urls_vector = src_urls

            if src_or_trg == "trg":
                docs_vector = trg_docs
                embedding_files_vector = trg_embedding_files
                urls_vector = trg_urls

            docs_vector.append(utils.expand_and_real_path_and_exists(line[0], raise_exception=True))
            embedding_files_vector.append(utils.expand_and_real_path_and_exists(line[1], raise_exception=False if generate_embeddings_arg else True))

            if (get_urls and line[2] == "-"):
                get_urls = False

            if get_urls:
                urls_vector.append(line[2])

        else:
            # Default format: src_doc\ttrg_doc\tsrc_embedding_file\ttrg_embedding_file\tsrc_url_or_dash\ttrg_url_or_dash

            if len(line) != 6:
                logging.warning(f"unexpected format in line #{idx + 1} (it will be skipped)")
                continue

            src_docs.append(utils.expand_and_real_path_and_exists(line[0], raise_exception=True))
            trg_docs.append(utils.expand_and_real_path_and_exists(line[1], raise_exception=True))
            src_embedding_files.append(utils.expand_and_real_path_and_exists(line[2], raise_exception=False if generate_embeddings_arg else True))
            trg_embedding_files.append(utils.expand_and_real_path_and_exists(line[3], raise_exception=False if generate_embeddings_arg else True))

            if (get_urls and (line[4] == "-" or line[5] == "-")):
                get_urls = False

            if get_urls:
                src_urls.append(line[4])
                trg_urls.append(line[5])

    if not get_urls:
        if len(src_urls) != 0:
            logging.warning(f"URLs were added to src URLs but are going to be ignored since not all of them were provided")
        if len(trg_urls) != 0:
            logging.warning(f"URLs were added to trg URLs but are going to be ignored since not all of them were provided")

        src_urls = [None] * len(src_docs)
        trg_urls = [None] * len(trg_docs)

    if file_open:
        data_src.close()

    return src_docs, trg_docs, src_embedding_files, trg_embedding_files, src_urls, trg_urls

def filter(src_embedding, trg_embedding, src_nolines=None, trg_nolines=None, percentage=0.3):
    if (src_nolines is not None and trg_nolines is not None):
        nolines_src = src_nolines
        nolines_trg = trg_nolines
    else:
        nolines_src = len(src_embedding)
        nolines_trg = len(trg_embedding)

    if (nolines_src == 0 or nolines_trg == 0):
        return True

    if (nolines_src >= 10 or nolines_trg >= 10):
        percentage_src = nolines_src / (nolines_src + nolines_trg)
        percentage_trg = nolines_trg / (nolines_src + nolines_trg)

        if max(percentage_src, percentage_trg) - min(percentage_src, percentage_trg) >= percentage:
            return True

    return False

def apply_mask(embeddings, mask, check_zeros_mask=False):
    for idx, embedding in enumerate(embeddings):
        if len(embedding.shape) == 1:
            axis = 0
            embeddings[idx] = embedding * mask
        elif len(embedding.shape) == 2:
            axis = 1

            for idx2, embedding2 in enumerate(embedding):
                embeddings[idx][idx2] = embedding2 * mask
        else:
            raise Exception("unexpected shape length: {len(embedding.shape)}")

        if check_zeros_mask:
            # Remove components of the embeddings where the mask value is 0.0
            embeddings[idx] = np.delete(embedding, mask <= sys.float_info.epsilon, axis=axis)

    return embeddings

def preprocess(src_docs, trg_docs, src_embeddings, trg_embeddings, src_embedding_files, trg_embedding_files, src_urls, trg_urls, **kwargs):
    # Apply heuristics in order to remove pairs which very likely will not be matches
    if ("apply_heuristics" in kwargs and kwargs["apply_heuristics"]):
        remove_idxs = []
        input_src_and_trg_splitted = kwargs["input_src_and_trg_splitted"] if ("input_src_and_trg_splitted" in kwargs) else False

        for idx, (src_embedding, trg_embedding) in enumerate(zip(src_embeddings, trg_embeddings)):
            if (not input_src_and_trg_splitted and filter(src_embedding, trg_embedding)):
                remove_idxs.insert(0, idx)

        remove_idxs_str = ' '.join(map(str, remove_idxs))
        logging.debug(f"elements filtered by heuristics (idxs): {'-' if remove_idxs_str == '' else remove_idxs_str}")

        for idx in remove_idxs:
            src_docs.pop(idx)
            trg_docs.pop(idx)
            src_embeddings.pop(idx)
            trg_embeddings.pop(idx)
            src_embedding_files.pop(idx)
            trg_embedding_files.pop(idx)
            src_urls.pop(idx)
            trg_urls.pop(idx)

    # Apply weights to embeddings
    if "weights_strategy" in kwargs:
        weights_strategy = kwargs["weights_strategy"]

        embeddings = weight_embeddings(src_embeddings + trg_embeddings, src_docs + trg_docs, weights_strategy=weights_strategy)

        if len(src_embeddings) + len(trg_embeddings) != len(embeddings):
            raise Exception(f"unexpected length after applying the weights. Expected length: {len(src_embeddings) + len(trg_embeddings)}. Actual length: {len(embeddings)}")

        src_embeddings = embeddings[:len(src_embeddings)]
        trg_embeddings = embeddings[len(src_embeddings):]

    # Merge embeddings
    if "merging_strategy" in kwargs:
        merging_strategy = kwargs["merging_strategy"]
        dim = kwargs["dim"] if "dim" in kwargs else DEFAULT_EMBEDDING_DIM

        if (not "do_not_merge_on_preprocessing" in kwargs or not kwargs["do_not_merge_on_preprocessing"]):
            # Merge src and trg embeddings
            for embeddings in (src_embeddings, trg_embeddings):
                for idx, embedding in enumerate(embeddings):
                    embeddings[idx] = merge_embedding(embedding, merging_strategy=merging_strategy, dim=dim)

    # Apply random mask to embeddings
    if ("random_mask_value" in kwargs and kwargs["random_mask_value"]):
        logging.info(f"using provided random mask")

        mask = kwargs["random_mask_value"].split(',')
        mask = np.float32(list(map(lambda x: np.float32(x), mask)))
        check_zeros_mask = False if "check_zeros_mask" not in kwargs else kwargs["check_zeros_mask"]

        # Check if the shape of the mask is correct
        for embeddings, label in ((src_embeddings, "source"), (trg_embeddings, "target")):
            embeddings_dim = len(embeddings[0])

            if "do_not_merge_on_preprocessing" in kwargs and kwargs["do_not_merge_on_preprocessing"]:
                embeddings_dim = len(embeddings[0][0])

            if (len(embeddings) != 0 and embeddings_dim != len(mask)):
                raise Exception(f"{label} embeddings shape and mask mismatch ({embeddings_dim} vs {len(mask)})")

        logging.debug(f"first elements of the provided mask ({min(len(mask), 5)} elements of {len(mask)}): {mask[0:min(len(mask), 5)]} ...")

        # Apply
        for embeddings in (src_embeddings, trg_embeddings):
            embeddings = apply_mask(embeddings, mask, check_zeros_mask=check_zeros_mask)

    return src_docs, trg_docs, src_embeddings, trg_embeddings, src_embedding_files, trg_embedding_files, src_urls, trg_urls

def get_faiss(src_docs, trg_docs, src_embeddings, trg_embeddings, take_knn=5, faiss_reverse_direction=False,
              dim=DEFAULT_EMBEDDING_DIM, threshold=None):
    results = []

    logging.info(f"dimensionality: {dim}")
    logging.info(f"using {take_knn} as neighbourhood size (knn)")

    # Create faiss index
    faiss_index = faiss.IndexFlatIP(dim)

    src_embedding_vectors = []
    trg_embedding_vectors = []

    for data, label in [(src_embeddings, "source"), (trg_embeddings, "target")]:
        if label == "source":
            embedding_vectors = src_embedding_vectors
        elif label == "target":
            embedding_vectors = trg_embedding_vectors
        else:
            raise Exception(f"unknown label: '{label}'")

        # Apply to src and trg docs and embeddings
#        for doc, embedding_file, embedding_data in data:
        for embedding_data in data:
            embedding = copy.copy(embedding_data)
            embedding = np.array(embedding)

            assert len(embedding.shape) == 1, f"The shape length of the {label} embedding must be 1, but is {len(embedding.shape)}"
            assert embedding.shape[0] == dim, f"The shape of the {label} embedding is {embedding.shape[0]}, but it must be {dim}"

            embedding_vectors.append(embedding)

    src_embedding_vectors = np.array(src_embedding_vectors)
    trg_embedding_vectors = np.array(trg_embedding_vectors)

    faiss.normalize_L2(src_embedding_vectors)
    faiss.normalize_L2(trg_embedding_vectors)

    if src_embedding_vectors.dtype == np.object:
        logging.warning(f"detected incorrect src embeddings (likely some of them were not correctly calculated)")
    if trg_embedding_vectors.dtype == np.object:
        logging.warning(f"detected incorrect trg embeddings (likely some of them were not correctly calculated)")

    faiss_index.add(src_embedding_vectors)

    D, I = faiss_index.search(trg_embedding_vectors, take_knn)

    # Get the best results
    for idx, i in enumerate(I):
        for idx2 in range(len(i)):
            try:
                # Check if the values obtained with FAISS are the expected (check out if they are valid values)
                trg_docs[idx]
                src_docs[i[idx2]]
            except IndexError as e:
                logging.warning(f"{str(e)} (skipping this result)")
                continue

            result_faiss = D[idx][idx2]

            if result_faiss >= 1.0 + sys.float_info.epsilon:
                logging.warning(f"faiss normalization not working well ({result_faiss})?")

            result_faiss = min(result_faiss, 1.0)

            results.append([i[idx2], idx, result_faiss])

    results.sort(key=itemgetter(-1), reverse=True)

    final_results = []
    already_src = set()
    already_trg = set()

    for r in results:
        if (r[1] in already_trg or r[0] in already_src):
            continue

        score = r[2]

        if (threshold is not None and score < threshold):
            continue

        if faiss_reverse_direction:
            final_results.append([trg_docs[r[1]], src_docs[r[0]], score])
        else:
            final_results.append([src_docs[r[0]], trg_docs[r[1]], score])

        already_src.add(r[0])
        already_trg.add(r[1])

    return final_results

def get_lev(src_embeddings, trg_embeddings, src_docs, trg_docs, noprocesses=0, noworkers=10, full=False):
    levenshtein_nfactor = levenshtein_norm_factor(src_embeddings, trg_embeddings)
    results_levenshtein = []

    # Multiprocessing
    if noprocesses > 0:
        logging.info(f"multiprocessing: using {noprocesses} processes and {noworkers} workers")

        src_idx = 0
        trg_idx = 0
        pool = Pool(processes=noprocesses)
        processes_args = []

        while True:
            # Get the args of the workers
            while src_idx < len(src_docs):
                while (trg_idx < len(trg_docs) and len(processes_args) < noworkers):
                    processes_args.append((src_embeddings[src_idx], trg_embeddings[trg_idx],
                                           levenshtein_nfactor, full
                                           [src_idx, trg_idx, src_idx * len(trg_docs) + trg_idx],))
                    trg_idx += 1

                if len(processes_args) >= noworkers:
                    break

                src_idx += 1
                trg_idx = 0

            # Check if we have finished
            if len(processes_args) == 0:
                break

            # Process the pool of workers
            results = pool.starmap(worker_lev, processes_args)

            del processes_args
            processes_args = []

            # Sort results by score
            results = sorted(results, key=lambda x: x[-1][-1])

            # Process results
            for r in results:
                src_doc = src_docs[r[-1][0]]
                trg_doc = trg_docs[r[-1][1]]

                if (r[0] is None or r[1] is None):
                    continue

                result_lev_sim = r[0]["similarity"]

                results_levenshtein.append([src_doc, trg_doc, result_lev_sim])

            del results

    # No multiprocessing
    else:
        lev_function = levenshtein.levenshtein if full else levenshtein.levenshtein_opt_space_and_band

        for embedding_src, src_doc in zip(src_embeddings, src_docs):
            for embedding_trg, trg_doc in zip(trg_embeddings, trg_docs):
                result_lev = lev_function(embedding_src, embedding_trg, nfactor=levenshtein_nfactor,
                                          diff_function_bool=lambda x, y: True,
                                          diff_function_value=lambda x, y: cosine_similarity(x, y))

                result_lev_sim = result_lev["similarity"]

                results_levenshtein.append([src_doc, trg_doc, result_lev_sim])

    return results_levenshtein

def get_distance(src_embeddings, trg_embeddings, src_docs, trg_docs, noprocesses=0, noworkers=10):
    results_avg = []

    # Multiprocessing
    if noprocesses > 0:
        logging.info(f"multiprocessing: using {noprocesses} processes and {noworkers} workers")

        src_idx = 0
        trg_idx = 0
        processes_args = []
        pool = Pool(processes=noprocesses)

        while True:
            # Get the args of the workers
            while src_idx < len(src_docs):
                while (trg_idx < len(trg_docs) and len(processes_args) < noworkers):
                    processes_args.append((src_embeddings[src_idx], trg_embeddings[trg_idx],
                                           [src_idx, trg_idx, src_idx * len(trg_docs) + trg_idx],))
                    trg_idx += 1

                if len(processes_args) >= noworkers:
                    break

                src_idx += 1
                trg_idx = 0

            # Check if we have finished
            if len(processes_args) == 0:
                break

            # Process the pool of workers
#            results = pool.starmap(worker_avg, processes_args)
            results = pool.starmap(worker_distance, processes_args)
            del processes_args
            processes_args = []

            # Sort results by score
            results = sorted(results, key=lambda x: x[-1][-1])

            # Process results
            for r in results:
                src_doc = src_docs[r[-1][0]]
                trg_doc = trg_docs[r[-1][1]]

                if (r[0] is None or r[1] is None):
                    continue

                result_avg = r[0]

                results_avg.append([src_doc, trg_doc, result_avg])

            del results

    # No multiprocessing
    else:
        storage = {}

        for embedding_src, src_doc in zip(src_embeddings, src_docs):
            for embedding_trg, trg_doc in zip(trg_embeddings, trg_docs):
#                result_avg = average_similarity(embedding_src, embedding_trg, storage=storage)
                cosine = cosine_similarity(embedding_src, embedding_trg)

                results_avg.append([src_doc, trg_doc, 1.0 - cosine])
#                results_avg.append([src_doc, trg_doc, result_avg])

    return results_avg

def main(args):
    dim = args.dim
    gold_standard = args.gold_standard
    noprocesses = args.processes
    noworkers = args.workers
    weights_strategy = args.weights_strategy
    merging_strategy = args.merging_strategy
    generate_embeddings_arg = args.generate_embeddings
    output_with_urls = args.output_with_urls
    gen_emb_optimization_strategy = None if args.gen_emb_optimization_strategy == 0 else args.gen_emb_optimization_strategy
    emb_optimization_strategy = None if args.emb_optimization_strategy == 0 else args.emb_optimization_strategy
    max_noentries = args.process_max_entries
    min_sanity_check = args.min_sanity_check if args.min_sanity_check >= 0 else 0
    random_mask_value = args.random_mask_value
    docalign_strategy = args.docalign_strategy
    results_strategy = args.results_strategy
    do_not_merge_on_preprocessing = args.do_not_merge_on_preprocessing
    generate_and_finish = args.generate_and_finish
    embed_script_path = args.embed_script_path

    # Configure logging
    utils.set_up_logging(level=args.logging_level, filename=args.log_file)

    # Process input file
    src_docs, trg_docs, src_embedding_files, trg_embedding_files, src_urls, trg_urls = \
        process_input_file(args, max_noentries)

    if weights_strategy != 0:
        logging.info(f"weights strategy: {weights_strategy}")

    logging.info(f"merging strategy: {merging_strategy}")

    if noworkers <= 0:
        noworkers = 10
        logging.warning(f"changing 'workers' value to {workers}. A non-valid value was provided")
    if not gold_standard:
        gold_standard = None

    # Generate embeddings (if needed)
    generate_embeddings(src_docs, src_embedding_files, args.src_lang, they_should_exist=not generate_embeddings_arg,
                        optimization_strategy=gen_emb_optimization_strategy, embed_script_path=embed_script_path)
    generate_embeddings(trg_docs, trg_embedding_files, args.trg_lang, they_should_exist=not generate_embeddings_arg,
                        optimization_strategy=gen_emb_optimization_strategy, embed_script_path=embed_script_path)

    if generate_and_finish:
        logging.info("the embeddings have been generated and the execution is going to finish")
        return

    src_embeddings = []
    trg_embeddings = []

    # Load embeddings
    for src_embedding in src_embedding_files:
        src_emb = get_embedding_vectors(src_embedding, dim=dim, optimization_strategy=emb_optimization_strategy)

        src_embeddings.append(src_emb)
    for trg_embedding in trg_embedding_files:
        trg_emb = get_embedding_vectors(trg_embedding, dim=dim, optimization_strategy=emb_optimization_strategy)

        trg_embeddings.append(trg_emb)

    # Sanity check: check if the embeddings have the expected shape
    for docs, embeddings, label in [(src_docs, src_embeddings, "source"), (trg_docs, trg_embeddings, "target")]:
        for idx, (doc, embedding) in enumerate(zip(docs[0:min(len(docs), min_sanity_check)],
                                                   embeddings[0:min(len(embeddings), min_sanity_check)])):
            nolines = utils.get_nolines(doc)

            if nolines == 0:
                logging.warning(f"file with 0 lines ({label} - {idx}): '{doc}'")
            if len(embedding.shape) != 2:
                raise Exception(f"unexpected shape of embedding ({label} - {idx}). Expected shape is 2: doc with sentences * dim. Actual shape: {len(embedding.shape)}")
            if embedding.shape[0] == 0:
                logging.warning(f"embedding with 0 elements ({label} - {idx})")
            if nolines != embedding.shape[0]:
                raise Exception(f"unexpected number of setences ({label} - {idx}). Expected sentences are {nolines}. Actual number of sentences: {embedding.shape[0]}")
            if embedding.shape[1] != dim:
                raise Exception(f"unexpected dimension of embedding ({label} - {idx}) according to the provided dim. Expected dim is {dim}. Actual dim: {embedding.shape[1]}")

    # Preprocess embeddings
    src_docs, trg_docs, src_embeddings, trg_embeddings, src_embedding_files, trg_embedding_files, src_urls, trg_urls = \
        preprocess(src_docs, trg_docs, src_embeddings, trg_embeddings, src_embedding_files, trg_embedding_files, src_urls, trg_urls,
                   weights_strategy=weights_strategy, merging_strategy=merging_strategy, random_mask_value=random_mask_value,
                   check_zeros_mask=args.check_zeros_mask, do_not_merge_on_preprocessing=do_not_merge_on_preprocessing,
                   apply_heuristics=args.apply_heuristics, input_src_and_trg_splitted=args.input_src_and_trg_splitted)

    if (len(src_embeddings) == 0 or len(trg_embeddings) == 0):
        logging.warning("there are not embeddings in both src and trg")
        return

    # Fix dim if necessary
    embeddings_dim = len(src_embeddings[0]) if len(src_embeddings) > 0 else None
    embeddings_dim = len(trg_embeddings[0]) if (len(trg_embeddings) > 0 and embeddings_dim is None) else embeddings_dim

    if do_not_merge_on_preprocessing:
        embeddings_dim = len(src_embeddings[0][0]) if (len(src_embeddings) > 0 and len(src_embeddings[0]) > 0) else None
        embeddings_dim = len(trg_embeddings[0][0]) if (len(trg_embeddings) > 0 and len(trg_embeddings[0]) > 0 and embeddings_dim is None) else embeddings_dim

    if (embeddings_dim is None and not generate_embeddings_arg):
        logging.warning(f"could not infer the dimension of the embeddings")
    elif (embeddings_dim != dim and embeddings_dim is not None):
        logging.info(f"dimension updated from {dim} to {embeddings_dim}")
        dim = embeddings_dim

    # Docalign, results and, optionally, evaluation
    if docalign_strategy == "faiss":
        faiss_reverse_direction = args.faiss_reverse_direction
        faiss_take_knn = args.faiss_take_knn

        faiss_args = [src_docs, trg_docs, src_embeddings, trg_embeddings]

        if faiss_reverse_direction:
            faiss_args.reverse()

        results_faiss = get_faiss(*faiss_args, take_knn=int(faiss_take_knn), faiss_reverse_direction=faiss_reverse_direction,
                                  dim=dim, threshold=args.faiss_threshold)
        urls_aligned_faiss = docalign(results_faiss, src_docs, trg_docs, src_urls, trg_urls, output_with_urls, only_docalign=True)

        # TODO filter here

        if results_strategy == 0:
            logging.info(f"results: get the best {faiss_take_knn} matches from src to trg docs, sort by score and do not select the either of the two docs again")

            for r in urls_aligned_faiss:
                print(f"{r[0]}\t{r[1]}")

            if gold_standard:
                recall, precision = evaluate.process_gold_standard(gold_standard, urls_aligned_faiss)
                print(f"recall, precision: {recall}, {precision}")

    elif docalign_strategy in ("lev", "lev-full"):
        results_lev = get_lev(src_embeddings, trg_embeddings, src_docs, trg_docs, noprocesses=noprocesses,
                              noworkers=noworkers, full=True if docalign_strategy == "lev-full" else False)

        urls_aligned_lev = docalign(results_lev, src_docs, trg_docs, src_urls, trg_urls, output_with_urls)
        union_and_int_lev = union_and_intersection(urls_aligned_lev) if urls_aligned_lev else None

        if results_strategy == 0:
            logging.info(f"results: union of best matches from src to trg and from trg to src")

            for r in union_and_int_lev["union"]:
                print(f"{r[0]}\t{r[1]}")

            if gold_standard:
                recall, precision = evaluate.process_gold_standard(gold_standard, union_and_int_lev["union"])
                print(f"recall, precision: {recall}, {precision}")

        elif results_strategy == 1:
            logging.info(f"results: intersection of best matches from src to trg and trg to src")

            for r in union_and_int_lev["intersection"]:
                print(f"{r[0]}\t{r[1]}")

            if gold_standard:
                recall, precision = evaluate.process_gold_standard(gold_standard, union_and_int_lev["intersection"])
                print(f"recall, precision: {recall}, {precision}")

    elif docalign_strategy == "just-merge":
        results_avg = get_distance(src_embeddings, trg_embeddings, src_docs, trg_docs, noprocesses=noprocesses, noworkers=noworkers)

        urls_aligned_avg = docalign(results_avg, src_docs, trg_docs, src_urls, trg_urls, output_with_urls)
        union_and_int_avg = union_and_intersection(urls_aligned_avg) if urls_aligned_avg else None

        if results_strategy == 0:
            logging.info(f"results: union of best matches from src to trg and from trg to src\n")

            for r in union_and_int_avg["union"]:
                print(f"{r[0]}\t{r[1]}")

            if gold_standard:
                recall, precision = evaluate.process_gold_standard(gold_standard, union_and_int_avg["union"])
                print(f"recall, precision: {recall}, {precision}")

        elif results_strategy == 1:
            logging.info(f"results: intersection of best matches from src to trg and trg to src")

            for r in union_and_int_avg["intersection"]:
                print(f"{r[0]}\t{r[1]}")

            if gold_standard:
                recall, precision = evaluate.process_gold_standard(gold_standard, union_and_int_avg["intersection"])
                print(f"recall, precision: {recall}, {precision}")
    else:
        raise Exception(f"unknown docalign strategy: '{docalign_strategy}'")

def check_args(args):
    if (args.generate_embeddings and args.gen_emb_optimization_strategy != args.emb_optimization_strategy):
        raise Exception("embeddings are going to be generated with an optimization strategy different from the one with they will be loaded (check --gen-emb-optimization-strategy and --emb-optimization-strategy)")
    if (args.generate_and_finish and not args.generate_embeddings):
        raise Exception("you cannot generate embeddings and finish the execution if you have not provided the flag in order to generate the embeddings")
    if (args.docalign_strategy in ("faiss", "just-merge") and (args.merging_strategy == 0 or args.do_not_merge_on_preprocessing)):
        raise Exception(f"docalign strategy '{args.docalign_strategy}' needs a merging strategy different of 0 and apply it on the preprocessing step (check --do-not-merge-on-preprocessing)")
    if (args.docalign_strategy in ("lev", "lev-full") and not args.do_not_merge_on_preprocessing):
        raise Exception(f"docalign strategy '{args.docalign_strategy}' needs to apply the merging strategy by itself instead of doing it on the on the preprocessing step (check --do-not-merge-on-preprocessing)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural document aligner')

    # Embedding
    parser.add_argument('input_file', metavar='input-file',
        help='TSV file with src_doc_path\\ttrg_doc_path\\tsrc_emb_path\\ttrg_emb_path\\tsrc_url_or_dash\\ttrg_url_or_dash entries and without header. The embeddings will be assumed to exists if --generate-embeddings is not provided. If --input-src-and-trg-splitted is provided, the expected format is doc_path\\temb_path\\turl_or_dash\\t<src|trg>')
    parser.add_argument('src_lang', metavar='src-lang',
        help='Source documents language')
    parser.add_argument('trg_lang', metavar='trg-lang',
        help='Target documents language')

    # Strategies
    parser.add_argument('--docalign-strategy', default='faiss',
        choices=['faiss', 'lev', 'lev-full', 'just-merge'],
        help='Document align strategy to get the pairs of documents aligned. Default is \'faiss\'')
    parser.add_argument('--weights-strategy', default=0, type=int,
        choices=range(0, 3 + 1),
        help='Indicates the strategy to follow in order to set weights to embeddings. Default is 0, which means do not apply any strategy')
    parser.add_argument('--merging-strategy', default=0, type=int,
        choices=range(0, 5 + 1),
        help='Indicates the strategy to follow in order to merge the embeddings. Default is 0, which means do not apply any strategy')
    parser.add_argument('--results-strategy', default=0, metavar='N', type=int,
        help='Indicates the strategy to follow in order to obtain the results of the pairs. Default is 0, which is the default strategy for the selected docalign strategy')
    parser.add_argument('--gen-emb-optimization-strategy', default=None, type=int,
        choices=range(0, 2 + 1),
        help='Optimization strategy of the embeddings which are going to be generated. The default value is do not apply any strategy')
    parser.add_argument('--emb-optimization-strategy', default=None, type=int,
        choices=range(0, 2 + 1),
        help='Optimization strategy of the embeddings which are going to be loaded. The default value is do not apply any strategy')

    # Multiprocessing
    parser.add_argument('--processes', default=0, metavar='N', type=int,
        help='Number of processes to use in order to parallelize')
    parser.add_argument('--workers', default=10, metavar='N', type=int,
        help='Number of workers to use in order to parallelize')

    # Embedding configuration
    parser.add_argument('--dim', default=DEFAULT_EMBEDDING_DIM, type=int, metavar='N',
        help='Dimensionality of the provided embeddings')
    parser.add_argument('--generate-embeddings', action="store_true",
        help='The embeddings provided in the input file will be generated. If the provided path of an embedding exists, it will be overwritten')
    parser.add_argument('--generate-and-finish', action="store_true",
        help='Generate the embeddings and finish the exit')
    parser.add_argument('--embed-script-path', metavar='PATH', default=f'{os.path.realpath(__file__).rpartition("/")[0]}/LASER/source/embed.py',
        help=f'Path to the script embed.py from LASER which will be used in order to generate the embeddings. The default value is \'{os.path.realpath(__file__).rpartition("/")[0]}/LASER/source/embed.py\'')
    parser.add_argument('--random-mask-value', metavar='<v_1>,<v_2>,...,<v_dim>',
        help='Random mask value. The expected format is: <value_1>,<value_2>,...,<value_n> (n=embeddings dim)')
    parser.add_argument('--check-zeros-mask', action="store_true",
        help='If --random-mask-value is provided beside this option, if any value of the mask is zero, the dimensionality will be reduced in that components')

    # Other
    parser.add_argument('--min-sanity-check', default=5, type=int, metavar='N',
        help='Min. quantity of documents to sanity check. Default is 5')
    parser.add_argument('--input-src-and-trg-splitted', action="store_true",
        help='If set, the expected format of the input file will be different from the initial and src and trg docs will be expected to have a line each instead of being specified in the same line')
    parser.add_argument('--do-not-merge-on-preprocessing', action="store_true",
        help='If set, the embeddings will not be merged on the preprocessing step and will be provided with the original length of the shape. This might be necessary for some docalign strategies which expect to apply some merging strategy instead of get the embeddings merged')
    parser.add_argument('--gold-standard', default=None, metavar='PATH',
        help='Path to the gold estandard. The expected format is src_doc_path\\ttrg_doc_path. If you want to use the provided URLs in the input file, use --output-with-urls and the URLs will be used instead of the paths')
    parser.add_argument('--apply-heuristics', action='store_true',
        help='Enable the heuristics to be applied')
    parser.add_argument('--output-with-urls', action="store_true",
        help='Generate the output with src and trg URLs instead of src and trg documents path. URLs have to be provided in the input file')
    parser.add_argument('--process-max-entries', metavar='N', default=None, type=int,
        help='Process only the first nth entries of the input file. The default value is process all entries')
    ## Faiss
    parser.add_argument('--faiss-threshold', type=float, metavar='F',
        help='Entries which have a value less than the threshold will not be added')
    parser.add_argument('--faiss-reverse-direction', action='store_true',
        help='Instead of index source docs and match with target docs, reverse the direction')
    parser.add_argument('--faiss-take-knn', default=5, metavar='N', type=int,
        help='Indicates the size of the neighbourhood when using faiss')
    ## Logging
    parser.add_argument('--logging-level', metavar='N', type=int, default=30,
                        help='Logging level. Default value is 10, which is WARNING')
    parser.add_argument('--log-file', metavar='PATH', default=None,
                        help='Log file where all the log entries will be stored')

    args = parser.parse_args()

    check_args(args)

    main(args)
