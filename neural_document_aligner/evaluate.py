
import os
import sys
import ast
import math
import logging
import argparse
import numpy as np

import levenshtein
import utils.utils as utils

def lev_opt(word1, word2, weight=1.0):
    m = np.zeros((2, len(word2) + 1))

    for idx1 in range(len(word1) + 1):
        for idx2 in range(m.shape[1]):
            if idx1 == 0:
                m[idx1][idx2] = idx2 * weight
                continue
            if idx2 == 0:
                m[1][idx2] = idx1 * weight
                continue

            diff = 0.0

            if word1[idx1 - 1] != word2[idx2 - 1]:
                diff = 1.0

            m[1][idx2] = min(min(m[1][idx2 - 1] + 1.0 * weight, m[1 - 1][idx2] + 1.0 * weight), m[1 - 1][idx2 - 1] + diff * weight)
        if idx1 != 0:
            m[0] = m[1]

    return m[-1][-1]

def lev_opt_space_and_band(word1, word2, weight=1.0, percentage=0.06, early_stopping=0.05):
    if (len(word1) < 10 or len(word2) < 10):
        return lev_opt(word1, word2, weight)

    m = np.ones((2, len(word2) + 1)) * np.inf
    perc_cols = math.floor((len(word1) + 1) * percentage)
    perc_rows = math.floor((len(word2) + 1) * percentage)

    for i in range(m.shape[1]):
        m[0][i] = i * weight

    idx1 = 0
    idx2 = 0
    max_idx1 = len(word1) + 1
    max_idx2 = m.shape[1]

    while idx1 < max_idx1:
        currently_iterated = idx1 / max_idx1 # [0, 1]
        idx2 = max(0, math.floor(currently_iterated * max_idx2) - perc_cols)
        idx2_limit = min(max_idx2, math.floor(currently_iterated * max_idx2) + perc_cols)

        if (idx1 and idx2):
            m[0][:idx2] = np.inf

        while idx2 < idx2_limit:
            if idx1 == 0:
                m[idx1][idx2] = idx2 * weight
                idx2 += 1
                continue
            if idx2 == 0:
                m[1][idx2] = idx1 * weight
                idx2 += 1
                continue

            diff = 0.0

            if word1[idx1 - 1] != word2[idx2 - 1]:
                diff = 1.0

            m[1][idx2] = min(min(m[1][idx2 - 1] + 1.0 * weight, m[1 - 1][idx2] + 1.0 * weight), m[1 - 1][idx2 - 1] + diff * weight)

            idx2 += 1

        if idx1:
            m[0] = m[1]

            if min(idx1 * weight, min(m[0])) > early_stopping:
                print(f"Early stopping: [{m[0][0]} ... {m[0][-1]}] (row: {idx1})")
                return 1.0 # Early stopping

        idx1 += 1

    print(f"[{m[-1][0]} ... {m[-1][-1]}]")

    return m[-1][-1]

def near_duplicates(url1, url2, docs_urls):
    if not docs_urls:
        return False

    doc1 = None
    doc2 = None

    try:
        doc1_idx = docs_urls['urls'].index(url1)
        doc1 = docs_urls['docs'][doc1_idx]
    except:
        logging.error(f"could not get url1 ({url1}) index because does not exist an asociated document")

    try:
        doc2_idx = docs_urls['urls'].index(url2)
        doc2 = docs_urls['docs'][doc2_idx]
    except:
        logging.error(f"could not get url2 ({url2}) index because does not exist an asociated document")

    if (not doc1 or not doc2):
        return False

    word1 = ""
    word2 = ""

    with open(doc1, "r") as doc1_file:
        for l in doc1_file:
            l = l.strip()
            word1 += f"{l} "
    with open(doc2, "r") as doc2_file:
        for l in doc2_file:
            l = l.strip()
            word2 += f"{l} "

    word1 = word1.rstrip()
    word2 = word2.rstrip()

    if lev_opt_space_and_band(word1, word2, 1 / max(len(word1), len(word2))) < 0.05:
        return True

    return False

def process_gold_standard(gs, result, filter=None, docs_urls=None, soft_recall=False):
    if len(result) == 0:
        logging.warning("there are no entries in the provided results")
        return 0.0, 0.0

    f = open(gs, "r")
    recall = 0.0
    precision = 0.0
    nolines = 0
    noresultsseen = 0
    filtered = 0

    with open(gs, "r") as f:
        for line in f:
            if (filter and filter not in line):
                filtered += 1
                continue

            urls = line.strip().split("\t")
            url1 = urls[0]
            url2 = urls[1]
            pair1 = (url1, url2)
            pair2 = (url2, url1)
            sr_res = set()
            nrs_found = False

            for r in result:
                if (r[0] == url1 or r[1] == url1):
                    if not nrs_found:
                        noresultsseen += 1
                        nrs_found = True
                    sr_res.add(r)

                if (r[0] == url2 or r[1] == url2):
                    if not nrs_found:
                        noresultsseen += 1
                        nrs_found = True
                    sr_res.add(r)

            if (pair1 in result or pair2 in result):
                logging.debug(f"the pair ('{url1}', '{url2}') is in the provided results")

                recall += 1
                precision += 1
            elif soft_recall:
                # Soft recall
                if (len(sr_res) == 1 and docs_urls is not None):
                    # (url1, url2) E gs
                    # sr_res[0] E results
                    if (sr_res[0][0] == url1 or sr_res[0][1] == url1):
                        # Check distance edition between url2 and 0 or 1
                        sr_res_compare = sr_res[0][0]

                        if sr_res[0][0] == url1:
                            sr_res_compare = sr_res[0][1]

                        # Check distance edition between url2 and sr_res_compare
                        if near_duplicates(url2, sr_res_compare, docs_urls):
                            logging.debug(f"the pair ('{url1}', '{url2}') is in our results (soft recall)")

                            recall += 1
                            precision += 1
                    else:
                        # Check distance edition between url1 and 0 or 1
                        sr_res_compare = sr_res[0][0]

                        if sr_res[0][0] == url2:
                            sr_res_compare = sr_res[0][1]

                        # Check distance edition between url1 and sr_res_compare
                        if near_duplicates(url2, sr_res_compare, docs_urls):
                            logging.debug(f"the pair ('{url1}', '{url2}') is in our results (soft recall)")

                            recall += 1
                            precision += 1
            nolines += 1

    logging.info(f"number of lines which have been used ({filtered} filtered from the total, which is {nolines + filtered}): {nolines}")
    logging.info(f"total results: {len(result)}")

    if nolines == 0:
        logging.warning("number of lines processed in gold standard is 0")
        return 0.0, 0.0

    recall /= float(nolines)

    if noresultsseen == 0:
        precision = 0.0
    else:
        precision /= noresultsseen

    return recall, precision

def process_results(path):
    file = open(path, "r")
    results = []

    for l in file:
        l = l.strip()
        pair = ast.literal_eval(l)

        results.append(pair)

    return results

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

            docs.append(line[0])
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

    return {'docs': docs, 'urls': urls}

def main(args):
    results = process_results(args.results)
    docs_urls = None

    if args.docs_urls_path:
        docs_urls = get_docs(args.docs_urls_path)

        if args.sanity_check:
            for r in results:
                url1 = r[0]
                url2 = r[1]

                if url1 not in docs_urls['urls']:
                    logging.warning(f"url (1) '{url1}' not found in provided file which should contain all the URLs")
                if url2 not in docs_urls['urls']:
                    logging.warning(f"url (2) '{url2}' not found in provided file which should contain all the URLs")

    r, p = process_gold_standard(args.gold, results, None, docs_urls)

    bn = os.path.basename(args.results)

    print(f"{len(bn) * '-'}--")
    print(f" {bn} ")
    print(f"{len(bn) * '-'}--")

    print(f"Recall: {r}")
    print(f"Precision: {p}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation with gold standard')

    parser.add_argument('results',
        help='Path to the document which contains the results')
    parser.add_argument('gold',
        help='Path to the Gold Standard document')

    parser.add_argument('--docs-urls-path',
        help='Path to the file which contains all the pairs with docs and urls relation')
    parser.add_argument('--sanity-check', action='store_true',
        help='Perform sanity check, what increases the time of execution')
    parser.add_argument('--logging-level', metavar='N', type=int, default=30,
                        help='Logging level. Default value is 30, which is WARNING')

    utils.set_up_logging(level=args.logging_level)

    args = parser.parse_args()

    main(args)
