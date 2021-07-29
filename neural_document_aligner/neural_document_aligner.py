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
import utils.embedding_util as embedding_util
import generate_embeddings as gen_embeddings
import evaluate
import levenshtein
from exceptions import FileFoundError
import constants

def generate_embeddings(docs, embedding_files, lang, they_should_exist=True, optimization_strategy=None, model=None,
                        max_mbytes_per_batch=constants.DEFAULT_MAX_MBYTES_PER_BATCH,
                        embeddings_batch_size=constants.DEFAULT_BATCH_SIZE, sentence_splitting=constants.DEFAULT_SENTENCE_SPLITTING):
    for idx, embedding in enumerate(embedding_files):
        # Check if the embeddings either should or not exist
        if os.path.isfile(embedding) != they_should_exist:
            if they_should_exist:
                logging.error(f"Embedding #{idx} should exist but it does not")
                raise FileNotFoundError(embedding)
            else:
                logging.error(f"Embedding #{idx} should not exist but it does")
                raise FileFoundError(embedding)

    if not they_should_exist:
        # Generate embeddings because they should not exist
        logging.info(f"Generating embeddings (batch size: {embeddings_batch_size})")

        if sentence_splitting:
            logging.info(f"Sentence splitting will be applied")

        gen_embeddings.process(docs, [lang] * len(docs), embedding_files, optimization_strategy=optimization_strategy,
                               model=model, max_mbytes_per_batch=max_mbytes_per_batch, batch_size=embeddings_batch_size,
                               sentence_splitting=sentence_splitting)

def get_embedding_vectors(embedding_file, dim=constants.DEFAULT_EMBEDDING_DIM, optimization_strategy=None):
    embedding = embedding_util.load(embedding_file, dim=dim, strategy=optimization_strategy)

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
        logging.warning("Empty or near-empty file -> not applying weights (i.e. weights = [1.0, ..., 1.0])")

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
        logging.warning(f"Different length of idf weights ({len(idf_weights)}) and sl weights ({len(sl_weights)})")

    return sl_weights * idf_weights

def get_weights_slidf_all(paths, cnt=len):
    results = []

    idf_weights = get_weights_idf_all(paths)
    sl_weights = []

    for p in paths:
        sl_weights.append(get_weights_sl(p, cnt))

    if len(idf_weights) != len(sl_weights):
        logging.warning(f"Different length of idf weights ({len(idf_weights)}) and sl weights ({len(sl_weights)})")

    for idx in range(len(sl_weights)):
        if len(sl_weights[idx]) != len(idf_weights[idx]):
            logging.warning(f"Different length of idf weights ({len(idf_weights)}) and sl weights ({len(sl_weights)}) for index {idx}, whose document should be '{paths[idx]}'")

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
        logging.warning(f"Shapes between embeddings and weights do not match ({len(embeddings)} vs {len(weights)}), so no weights are going to be applied")
    else:
        # Apply weights
        for emb_idx, (embedding, weight) in enumerate(zip(embeddings, weights)):
            embs = np.array(embedding, dtype=np.float32)
            weight = np.array(weight)

            if embs.shape[0] != weight.shape[0]:
                logging.warning(f"Shapes between embedding and weights do not match ({embs.shape[0]} vs {weight.shape[0]}), so no weights are going to be applied to the current embedding (idx {emb_idx})")
                continue

            for idx in range(len(embs)):
                embs[idx] *= weight[idx]

            embeddings[emb_idx] = embs

    return embeddings

def merge_embedding(doc_embeddings, merging_strategy=0, dim=constants.DEFAULT_EMBEDDING_DIM):
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

def max_split3_embedding(embedding, dim=constants.DEFAULT_EMBEDDING_DIM):
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
        logging.warning(f"Unexpected size of result in max_split3_embedding ({len(result)})")

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

def worker_distance(embedding_src, embedding_trg, return_value):
    cosine = cosine_similarity(avg_embedding_src_vector, avg_embedding_trg_vector)

    return (1.0 - cosine, return_value)

def docalign(results, src_docs, trg_docs, src_urls, trg_urls, output_with_urls, only_docalign=False):
    result = {'src': {}, 'trg': {}}
    scores = {}

    if only_docalign:
        result = set()

        for r in results:
            src_doc = r[0]
            trg_doc = r[1]
            score = r[-1]

            if output_with_urls:
                src_idx = src_docs.index(src_doc)
                src_doc = src_urls[src_idx]

                trg_idx = trg_docs.index(trg_doc)
                trg_doc = trg_urls[trg_idx]

            result.add((src_doc, trg_doc))
            scores[hash(src_doc) + hash(trg_doc)] = score

        return result, scores

    if (len(results) == 0 or len(results[0]) == 0):
        return result, scores

    # Get best result for src and trg docs
    for r in results:
        src_doc = r[0]
        trg_doc = r[1]
        score = r[-1]

        if output_with_urls:
            src_idx = src_docs.index(src_doc)
            src_doc = src_urls[src_idx]

            trg_idx = trg_docs.index(trg_doc)
            trg_doc = trg_urls[trg_idx]

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

        scores[hash(src_doc) + hash(trg_doc)] = score

    return result, scores

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

def process_input_file(args, max_noentries=None):
    input_file = args.input_file
    generate_embeddings_arg = args.generate_embeddings

    src_docs, trg_docs = [], []
    src_embedding_files, trg_embedding_files = [], []
    src_urls, trg_urls = [], []
    get_urls = True
    get_docs = True
    file_open = False

    if input_file == "-":
        data_src = sys.stdin.readlines()
    else:
        data_src = open(input_file, "r", encoding="utf-8")
        file_open = True

    for idx, line in enumerate(data_src):
        if (max_noentries and idx >= max_noentries):
            logging.debug(f"Max. number of lines to process from the input file has been reached ({idx} lines processed from '{input_file}')")
            break

        line = line.strip().split("\t")

        # Expected format: doc\tembedding_file\turl_or_dash\t<src|trg>

        if len(line) != 4:
            logging.warning(f"Unexpected format in line #{idx + 1} (it will be skipped)")
            continue

        src_or_trg = line[3].lower()

        if src_or_trg not in ("src", "trg"):
            logging.warning(f"Unexpected format in line #{idx + 1} (it will be skipped)")
            continue

        docs_vector = src_docs
        embedding_files_vector = src_embedding_files
        urls_vector = src_urls

        if src_or_trg == "trg":
            docs_vector = trg_docs
            embedding_files_vector = trg_embedding_files
            urls_vector = trg_urls

        # Optional documents
        if (get_docs and line[0] == "-"):
            get_docs = False
        if get_docs:
            docs_vector.append(utils.expand_and_real_path_and_exists(line[0], raise_exception=True))

        # Embedding path
        embedding_files_vector.append(utils.expand_and_real_path_and_exists(line[1], raise_exception=False if generate_embeddings_arg else True))

        # Optional URLs
        if (get_urls and line[2] == "-"):
            get_urls = False
        if get_urls:
            urls_vector.append(line[2])

    if (not get_docs and not get_urls):
        raise Exception("it is necessary to provide either documents paths or URLs paths (or both), but neither were provided")

    if not get_docs:
        if len(src_docs) != 0:
            logging.warning(f"docs were added to src docs but are going to be ignored since not all of them were provided")
        if len(trg_docs) != 0:
            logging.warning(f"docs were added to trg docs but are going to be ignored since not all of them were provided")

        src_docs = [None] * len(src_urls)
        trg_docs = [None] * len(trg_urls)

    if not get_urls:
        if len(src_urls) != 0:
            logging.warning(f"URLs were added to src URLs but are going to be ignored since not all of them were provided")
        if len(trg_urls) != 0:
            logging.warning(f"URLs were added to trg URLs but are going to be ignored since not all of them were provided")

        src_urls = [None] * len(src_docs)
        trg_urls = [None] * len(trg_docs)

    if (len(src_docs) != len(src_embedding_files) or len(src_docs) != len(src_urls)):
        raise Exception("unexpected size of the src data")
    if (len(trg_docs) != len(trg_embedding_files) or len(trg_docs) != len(trg_urls)):
        raise Exception("unexpected size of the trg data")

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

def preprocess(src_docs, trg_docs, src_embeddings, trg_embeddings, **kwargs):
    # Apply weights to embeddings
    if "weights_strategy" in kwargs:
        weights_strategy = kwargs["weights_strategy"]

        embeddings = weight_embeddings(src_embeddings + trg_embeddings, src_docs + trg_docs, weights_strategy=weights_strategy)

        if len(src_embeddings) + len(trg_embeddings) != len(embeddings):
            raise Exception(f"unexpected length after applying the weights. Expected length: {len(src_embeddings) + len(trg_embeddings)}. Actual length: {len(embeddings)}")

        src_embeddings[0:] = embeddings[:len(src_embeddings)]
        trg_embeddings[0:] = embeddings[len(src_embeddings):]

    # Merge embeddings
    if "merging_strategy" in kwargs:
        merging_strategy = kwargs["merging_strategy"]
        dim = kwargs["dim"] if "dim" in kwargs else constants.DEFAULT_EMBEDDING_DIM

        if (not "do_not_merge_on_preprocessing" in kwargs or not kwargs["do_not_merge_on_preprocessing"]):
            # Merge src and trg embeddings
            for embeddings in (src_embeddings, trg_embeddings):
                for idx, embedding in enumerate(embeddings):
                    embeddings[idx] = merge_embedding(embedding, merging_strategy=merging_strategy, dim=dim)

    # Apply random mask to embeddings
    if ("mask_value" in kwargs and kwargs["mask_value"]):
        logging.info(f"Using provided random mask")

        mask = kwargs["mask_value"].split(',')
        mask = np.float32(list(map(lambda x: np.float32(x), mask)))
        check_zeros_mask = False if "check_zeros_mask" not in kwargs else kwargs["check_zeros_mask"]

        # Check if the shape of the mask is correct
        for embeddings, label in ((src_embeddings, "source"), (trg_embeddings, "target")):
            embeddings_dim = len(embeddings[0])

            if "do_not_merge_on_preprocessing" in kwargs and kwargs["do_not_merge_on_preprocessing"]:
                embeddings_dim = len(embeddings[0][0])

            if (len(embeddings) != 0 and embeddings_dim != len(mask)):
                raise Exception(f"{label} embeddings shape and mask mismatch ({embeddings_dim} vs {len(mask)})")

        logging.debug(f"First elements of the provided mask ({min(len(mask), 5)} elements of {len(mask)}): {mask[0:min(len(mask), 5)]} ...")

        # Apply
        for embeddings in (src_embeddings, trg_embeddings):
            embeddings[0:] = apply_mask(embeddings, mask, check_zeros_mask=check_zeros_mask)

    return src_docs, trg_docs, src_embeddings, trg_embeddings

def get_faiss(src_docs, trg_docs, src_embeddings, trg_embeddings, take_knn=5, faiss_reverse_direction=False,
              dim=constants.DEFAULT_EMBEDDING_DIM, threshold=None):
    results = []

    logging.info(f"Dimensionality: {dim}")
    logging.info(f"Using {take_knn} as neighbourhood size (knn)")

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

    if src_embedding_vectors.dtype == object:
        logging.warning(f"Detected incorrect src embeddings (likely some of them were not correctly calculated)")
    if trg_embedding_vectors.dtype == object:
        logging.warning(f"Detected incorrect trg embeddings (likely some of them were not correctly calculated)")

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
                logging.warning(f"Skipping this result: {str(e)}")
                continue

            result_faiss = D[idx][idx2]

            if result_faiss >= 1.0 + sys.float_info.epsilon:
                logging.warning(f"Faiss normalization not working well ({result_faiss})?")

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

def get_lev(src_embeddings, trg_embeddings, src_docs, trg_docs, noprocesses=0, noworkers=10, full=False,
            threshold=None, apply_heuristics=False):
    levenshtein_nfactor = levenshtein_norm_factor(src_embeddings, trg_embeddings)
    results_levenshtein = []

    # Multiprocessing
    if noprocesses > 0:
        logging.info(f"Multiprocessing: using {noprocesses} processes and {noworkers} workers")

        src_idx = 0
        trg_idx = 0
        pool = Pool(processes=noprocesses)
        processes_args = []

        while True:
            # Get the args of the workers
            while src_idx < len(src_docs):
                while (trg_idx < len(trg_docs) and len(processes_args) < noworkers):
                    if (not apply_heuristics or not filter(src_embeddings[src_idx], trg_embeddings[trg_idx])):
                        processes_args.append((src_embeddings[src_idx], trg_embeddings[trg_idx],
                                               levenshtein_nfactor, full,
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

                if (threshold is not None and result_lev_sim < threshold):
                    continue

                results_levenshtein.append([src_doc, trg_doc, result_lev_sim])

            del results

    # No multiprocessing
    else:
        lev_function = levenshtein.levenshtein if full else levenshtein.levenshtein_opt_space_and_band

        for embedding_src, src_doc in zip(src_embeddings, src_docs):
            for embedding_trg, trg_doc in zip(trg_embeddings, trg_docs):
                if (apply_heuristics and filter(embedding_src, embedding_trg)):
                    continue

                result_lev = lev_function(embedding_src, embedding_trg, nfactor=levenshtein_nfactor,
                                          diff_function_bool=lambda x, y: True,
                                          diff_function_value=lambda x, y: cosine_similarity(x, y))

                result_lev_sim = result_lev["similarity"]

                if (threshold is not None and result_lev_sim < threshold):
                    continue

                results_levenshtein.append([src_doc, trg_doc, result_lev_sim])

    return results_levenshtein

def get_distance(src_embeddings, trg_embeddings, src_docs, trg_docs, noprocesses=0, noworkers=10, threshold=None,
                 apply_heuristics=False):
    results_distance = []

    # Multiprocessing
    if noprocesses > 0:
        logging.info(f"Multiprocessing: using {noprocesses} processes and {noworkers} workers")

        src_idx = 0
        trg_idx = 0
        processes_args = []
        pool = Pool(processes=noprocesses)

        while True:
            # Get the args of the workers
            while src_idx < len(src_docs):
                while (trg_idx < len(trg_docs) and len(processes_args) < noworkers):
                    if (not apply_heuristics or not filter(src_embeddings[src_idx], trg_embeddings[trg_idx])):
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

                result_distance = r[0]

                if (threshold is not None and result_distance < threshold):
                    continue

                results_distance.append([src_doc, trg_doc, result_distance])

            del results

    # No multiprocessing
    else:
        for embedding_src, src_doc in zip(src_embeddings, src_docs):
            for embedding_trg, trg_doc in zip(trg_embeddings, trg_docs):
                if (apply_heuristics and filter(embedding_src, embedding_trg)):
                    continue

                cosine = cosine_similarity(embedding_src, embedding_trg)

                if (threshold is not None and 1.0 - cosine < threshold):
                    continue

                results_distance.append([src_doc, trg_doc, 1.0 - cosine])

    return results_distance

def docalign_strategy_applies_own_embedding_merging(docalign_strategy):
    if args.docalign_strategy in ("faiss", "just-merge"):
        return False
    elif args.docalign_strategy in ("lev", "lev-full"):
        return True

    raise Exception(f"unknown docalign strategy: '{docalign_strategy}'")

def main(args):
    # Args
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
    mask_value = args.mask_value
    docalign_strategy = args.docalign_strategy
    results_strategy = args.results_strategy
    generate_and_finish = args.generate_and_finish
    model = args.model
    max_mbytes_per_batch = args.max_mbytes_per_batch
    max_loaded_sent_embs_at_once = args.max_loaded_sent_embs_at_once
    apply_heuristics = args.apply_heuristics
    threshold = args.threshold
    embeddings_batch_size = args.embeddings_batch_size
    do_not_show_scores = args.do_not_show_scores
    output_with_idxs = args.output_with_idxs
    sentence_splitting = args.sentence_splitting
    # Not args
    do_not_merge_on_preprocessing = docalign_strategy_applies_own_embedding_merging(docalign_strategy)
    docs_were_not_provided = False

    # Configure logging
    utils.set_up_logging(level=args.logging_level, filename=args.log_file, display_when_file=args.log_display)

    # Process input file
    src_docs, trg_docs, src_embedding_files, trg_embedding_files, src_urls, trg_urls = \
        process_input_file(args, max_noentries)

    # Check if the documents' path were provided
    if ((len(src_docs) != 0 and src_docs[0] is None) or
        (len(trg_docs) != 0 and trg_docs[0] is None)):
        if (generate_embeddings_arg or not output_with_urls or (weights_strategy is not None and weights_strategy != 0)):
            raise Exception("when you do not provide documents' paths you cannot: set '--generate-embeddings', do not set '--output-with-urls' or set '--weights-strategy' with a value different of 0")

        docs_were_not_provided = True
        output_with_urls = False # We force the user to set this flag to let him know about the behaviour, but we disable it internally

    if docs_were_not_provided:
        logging.info("Since documents' paths were not provided, URLs will be used instead as if they were the documents' paths")

    if weights_strategy != 0:
        logging.info(f"Weights strategy: {weights_strategy}")

    logging.info(f"Merging strategy: {merging_strategy}")

    if noworkers <= 0:
        noworkers = 10
        logging.warning(f"Changing 'workers' value to {workers} (a non-valid value was provided)")
    if not gold_standard:
        gold_standard = None

    # Generate embeddings (if needed)
    generate_embeddings(src_docs, src_embedding_files, args.src_lang, they_should_exist=not generate_embeddings_arg,
                        optimization_strategy=gen_emb_optimization_strategy, model=model, max_mbytes_per_batch=max_mbytes_per_batch,
                        embeddings_batch_size=embeddings_batch_size, sentence_splitting=sentence_splitting)
    generate_embeddings(trg_docs, trg_embedding_files, args.trg_lang, they_should_exist=not generate_embeddings_arg,
                        optimization_strategy=gen_emb_optimization_strategy, model=model, max_mbytes_per_batch=max_mbytes_per_batch,
                        embeddings_batch_size=embeddings_batch_size, sentence_splitting=sentence_splitting)

    if generate_and_finish:
        logging.info("The embeddings have been generated and the execution is going to finish")
        return

    src_embeddings = []
    trg_embeddings = []
    start_idx = 0
    end_idx = 0

    # Get document-level embeddings from sentence-level embeddings (batches)
    while end_idx < len(src_embedding_files):
        start_idx = end_idx
        end_idx = min(start_idx + max_loaded_sent_embs_at_once, len(src_embedding_files))

        logging.debug(f"Loading embeddings from {start_idx} to {end_idx}")

        # Load embeddings
        for src_embedding in src_embedding_files[start_idx:end_idx]:
            src_emb = get_embedding_vectors(src_embedding, dim=dim, optimization_strategy=emb_optimization_strategy)

            src_embeddings.append(src_emb)
        for trg_embedding in trg_embedding_files[start_idx:end_idx]:
            trg_emb = get_embedding_vectors(trg_embedding, dim=dim, optimization_strategy=emb_optimization_strategy)

            trg_embeddings.append(trg_emb)

        # Sanity check: check if the embeddings have the expected shape
        if (start_idx < min_sanity_check):
            for docs, embeddings, label in [(src_docs, src_embeddings, "source"), (trg_docs, trg_embeddings, "target")]:
                for idx, (doc, embedding) in enumerate(zip(docs[start_idx:min(len(docs), min_sanity_check, end_idx)],
                                                           embeddings[start_idx:min(len(embeddings), min_sanity_check, end_idx)])):
                    if doc is None:
                        nolines = embedding.shape[0] if embedding.shape[0] != 0 else 1
                    else:
                        nolines = utils.get_nolines(doc)

                    if nolines == 0:
                        logging.warning(f"File with 0 lines ({label} - {idx}): '{doc}'")
                    if len(embedding.shape) != 2:
                        raise Exception(f"unexpected shape of embedding ({label} - {idx}). Expected shape is 2: doc with sentences * dim. Actual shape: {len(embedding.shape)}")
                    if embedding.shape[0] == 0:
                        logging.warning(f"Embedding with 0 elements ({label} - {idx})")
                    # Sentence splitting might do that the length of the embedding is not the expected from the number of lines of the document
                    #if nolines != embedding.shape[0]:
                    #    raise Exception(f"unexpected number of setences ({label} - {idx}). Expected sentences are {nolines}. Actual number of sentences: {embedding.shape[0]}")
                    if embedding.shape[1] != dim:
                        raise Exception(f"unexpected dimension of embedding ({label} - {idx}) according to the provided dim. Expected dim is {dim}. Actual dim: {embedding.shape[1]}")

        # Preprocess embeddings
        src_docs[start_idx:end_idx], trg_docs[start_idx:end_idx], src_embeddings[start_idx:end_idx], trg_embeddings[start_idx:end_idx] = \
            preprocess(src_docs[start_idx:end_idx], trg_docs[start_idx:end_idx], src_embeddings[start_idx:end_idx], trg_embeddings[start_idx:end_idx],
                       weights_strategy=weights_strategy, merging_strategy=merging_strategy, mask_value=mask_value,
                       check_zeros_mask=args.check_zeros_mask, do_not_merge_on_preprocessing=do_not_merge_on_preprocessing)

    if (len(src_embeddings) == 0 or len(trg_embeddings) == 0):
        logging.warning("There are not embeddings in both src and trg")
        return

    # Fix dim if necessary
    embeddings_dim = len(src_embeddings[0]) if len(src_embeddings) > 0 else None
    embeddings_dim = len(trg_embeddings[0]) if (len(trg_embeddings) > 0 and embeddings_dim is None) else embeddings_dim

    if do_not_merge_on_preprocessing:
        embeddings_dim = len(src_embeddings[0][0]) if (len(src_embeddings) > 0 and len(src_embeddings[0]) > 0) else None
        embeddings_dim = len(trg_embeddings[0][0]) if (len(trg_embeddings) > 0 and len(trg_embeddings[0]) > 0 and embeddings_dim is None) else embeddings_dim

    if (embeddings_dim is None and not generate_embeddings_arg):
        logging.warning(f"Could not infer the dimension of the embeddings")
    elif (embeddings_dim != dim and embeddings_dim is not None):
        logging.info(f"Dimension updated from {dim} to {embeddings_dim}")
        dim = embeddings_dim

    if threshold is not None:
        logging.info(f"Using threshold: {threshold}")

    results_variable = None

    # Docalign, results and, optionally, evaluation
    if docalign_strategy == "faiss":
        faiss_reverse_direction = args.faiss_reverse_direction
        faiss_take_knn = args.faiss_take_knn

        faiss_args = [src_docs, trg_docs, src_embeddings, trg_embeddings]

        if docs_were_not_provided:
            faiss_args = [src_urls, trg_urls, src_embeddings, trg_embeddings]

        if faiss_reverse_direction:
            faiss_args.reverse()

        results_faiss = get_faiss(*faiss_args, take_knn=int(faiss_take_knn), faiss_reverse_direction=faiss_reverse_direction,
                                  dim=dim, threshold=threshold)
        urls_aligned_faiss, scores = docalign(results_faiss, src_docs, trg_docs, src_urls, trg_urls, output_with_urls, only_docalign=True)

        if results_strategy == 0:
            logging.info(f"Results: get the best {faiss_take_knn} matches from src to trg docs, sort by score and do not select the either of the two docs again")

            results_variable = urls_aligned_faiss

    elif docalign_strategy in ("lev", "lev-full"):
        results_lev = get_lev(src_embeddings, trg_embeddings, src_docs, trg_docs, noprocesses=noprocesses,
                              noworkers=noworkers, full=True if docalign_strategy == "lev-full" else False,
                              apply_heuristics=apply_heuristics, threshold=threshold)

        urls_aligned_lev, scores = docalign(results_lev, src_docs, trg_docs, src_urls, trg_urls, output_with_urls)
        union_and_int_lev = union_and_intersection(urls_aligned_lev) if urls_aligned_lev else None

        if results_strategy == 0:
            logging.info(f"Results: union of best matches from src to trg and from trg to src")

            results_variable = union_and_int_lev["union"]

        elif results_strategy == 1:
            logging.info(f"Results: intersection of best matches from src to trg and trg to src")

            results_variable = union_and_int_lev["intersection"]

    elif docalign_strategy == "just-merge":
        results_distance = get_distance(src_embeddings, trg_embeddings, src_docs, trg_docs, noprocesses=noprocesses, noworkers=noworkers,
                                        apply_heuristics=apply_heuristics, threshold=threshold)

        urls_aligned_distance, scores = docalign(results_distance, src_docs, trg_docs, src_urls, trg_urls, output_with_urls)
        union_and_int_distance = union_and_intersection(urls_aligned_distance) if urls_aligned_distance else None

        if results_strategy == 0:
            logging.info(f"Results: union of best matches from src to trg and from trg to src\n")

            results_variable = union_and_int_distance["union"]

        elif results_strategy == 1:
            logging.info(f"Results: intersection of best matches from src to trg and trg to src")

            results_variable = union_and_int_distance["intersection"]

    else:
        raise Exception(f"unknown docalign strategy: '{docalign_strategy}'")

    if results_variable is None:
        raise Exception("could not get the results (maybe wrong results strategy?)")

    # Print results
    for r in results_variable:
        src_result = r[0]
        trg_result = r[1]
        score = "unknown"
        hash_score = hash(r[0]) + hash(r[1])

        if hash_score in scores.keys():
            score = scores[hash_score]

        # Use indexes?
        if output_with_idxs:
            if (docs_were_not_provided or output_with_urls):
                # The results contain URLs
                src_result = src_urls.index(src_result)
                trg_result = trg_urls.index(trg_result)
            else:
                # The results contain documents paths
                src_result = src_docs.index(src_result)
                trg_result = trg_docs.index(trg_result)

        if do_not_show_scores:
            print(f"{src_result}\t{trg_result}")
        else:
            print(f"{src_result}\t{trg_result}\t{score}")

    # Evaluation
    if gold_standard:
        recall, precision = evaluate.process_gold_standard(gold_standard, results_variable)
        print(f"recall, precision: {recall}, {precision}")

def check_args(args):
    if (args.generate_embeddings and args.gen_emb_optimization_strategy != args.emb_optimization_strategy):
        raise Exception("embeddings are going to be generated with an optimization strategy different from the one with they will be loaded (check --gen-emb-optimization-strategy and --emb-optimization-strategy)")
    if (args.generate_and_finish and not args.generate_embeddings):
        raise Exception("you cannot generate embeddings and finish the execution if you have not provided the flag in order to generate the embeddings")
    if (not docalign_strategy_applies_own_embedding_merging(args.docalign_strategy) and args.merging_strategy == 0):
        raise Exception(f"docalign strategy '{args.docalign_strategy}' needs a merging strategy different of 0")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural document aligner')

    # Embedding
    parser.add_argument('input_file', metavar='input-file',
        help='TSV file with doc_path_or_dash\\temb_path\\turl_or_dash\\t<src|trg> entries and without header. The embeddings will be assumed to exists if --generate-embeddings is not set. Either documents or URLs have to be provided (or both); if documents\' paths are not provided, you will not be able to: generate embeddings, get the output with the documents\' paths or apply a weight strategy')
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
    parser.add_argument('--model', metavar='MODEL', default=None,
        help=f'Model to use from \'sentence_transformers\'. The default model is \'{constants.DEFAULT_ST_MODEL}\'')
    parser.add_argument('--dim', default=constants.DEFAULT_EMBEDDING_DIM, type=int, metavar='N',
        help='Dimensionality of the provided embeddings')
    parser.add_argument('--generate-embeddings', action="store_true",
        help='The embeddings provided in the input file will be generated. If the provided path of an embedding exists, it will be overwritten')
    parser.add_argument('--max-mbytes-per-batch', default=constants.DEFAULT_MAX_MBYTES_PER_BATCH, type=int, metavar='N',
        help=f'Max. MB which will be used per batch when generating embeddings (the size is not guaranteed). The default value is {constants.DEFAULT_MAX_MBYTES_PER_BATCH}')
    parser.add_argument('--embeddings-batch-size', default=constants.DEFAULT_BATCH_SIZE, type=int, metavar='N',
        help=f'Batch size for the embeddings generation. The default value is {constants.DEFAULT_BATCH_SIZE}')
    parser.add_argument('--generate-and-finish', action="store_true",
        help='Generate the embeddings and finish the exit')
    parser.add_argument('--mask-value', metavar='<v_1>,<v_2>,...,<v_dim>',
        help='Mask value which will be applied to every embedding. The expected format is: <value_1>,<value_2>,...,<value_n> (n=embeddings dim)')
    parser.add_argument('--check-zeros-mask', action="store_true",
        help='If --random-mask-value is provided beside this option, if any value of the mask is zero, the dimensionality will be reduced in that components')

    # Other
    parser.add_argument('--min-sanity-check', default=5, type=int, metavar='N',
        help='Min. quantity of documents to sanity check. Default is 5')
    parser.add_argument('--sentence-splitting', action="store_true",
        help='Apply sentence splitting to the documents before generating the embeddings')
    parser.add_argument('--do-not-show-scores', action="store_true",
        help='If set, the scores of the matches will not be shown')
    parser.add_argument('--threshold', type=float, metavar='F', default=None,
        help='Matches with score less than the provided threshold will not be added')
    parser.add_argument('--gold-standard', default=None, metavar='PATH',
        help='Path to the gold estandard. The expected format is src_doc_path\\ttrg_doc_path. If you want to use the provided URLs in the input file, use --output-with-urls and the URLs will be used instead of the paths')
    parser.add_argument('--apply-heuristics', action='store_true',
        help='Enable the heuristics to be applied')
    parser.add_argument('--output-with-urls', action="store_true",
        help='Generate the output with src and trg URLs instead of src and trg documents path. URLs have to be provided in the input file')
    parser.add_argument('--output-with-idxs', action="store_true",
        help='Generate the output with src and trg indexes instead of src and trg documents path')
    parser.add_argument('--max-loaded-sent-embs-at-once', metavar='N', default=1000, type=int,
        help='The sentence-level embeddings have to be loaded in memory, but this might be a problem if there is not sufficient memory available. With this option, the quantity of sentence-level embeddings loaded in memory at once can be configured in order to avoid to get run out of memory (once this embeddings have been loaded, they will become into document-level embeddings). The default value is 1000')
    parser.add_argument('--process-max-entries', metavar='N', default=None, type=int,
        help='Process only the first nth entries of the input file. The default value is process all entries')
    ## Faiss
    parser.add_argument('--faiss-reverse-direction', action='store_true',
        help='Instead of index source docs and match with target docs, reverse the direction')
    parser.add_argument('--faiss-take-knn', default=5, metavar='N', type=int,
        help='Indicates the size of the neighbourhood when using faiss')
    ## Logging
    parser.add_argument('--logging-level', metavar='N', type=int, default=constants.DEFAULT_LOGGING_LEVEL,
                        help=f'Logging level. Default value is {constants.DEFAULT_LOGGING_LEVEL}')
    parser.add_argument('--log-file', metavar='PATH', default=None,
                        help='Log file where all the log entries will be stored')
    parser.add_argument('--log-display', action='store_true',
                        help='If you set --log-file, logging messages will still be stored but not displayed to standar error output. With this option, the messages will be stored in the log file and also will be displayed')

    args = parser.parse_args()

    check_args(args)

    try:
        main(args)
    except MemoryError as e:
        logging.critical("You ran out of memory (--max-loaded-sent-embs-at-once might be a solution)")
        raise e
