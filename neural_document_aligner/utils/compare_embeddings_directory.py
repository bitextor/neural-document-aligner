
from os import listdir
import sys
import argparse
import logging

import numpy as np

import compare_embeddings
import utils

DEFAULT_EMBEDDING_DIM = 1024

def main(args):
    dir1 = args.embedding_dir_1
    fsp1 = args.lang_position_1
    opt1 = args.embedding_opt_1
    dir2 = args.embedding_dir_2
    fsp2 = args.lang_position_2
    opt2 = args.embedding_opt_2
    lsep = args.lang_sep
    cont = args.cont_when_match
    perc1 = args.percentage_dir_1
    perc2 = args.percentage_dir_2
    dim = args.dim

    if (perc >= 0.0 and perc <= 1.0 and perc2 >= 0.0 and perc2 <= 1.0):
        raise Exception("the percentages have to be values in the range [0.0, 1.0]")

    if perc2 < 1.0:
        logging.warning(f"using {perc2 * 100.0:.2f}% of the total samples, what means that if the order is not the same in both sides, likely the recall will not be 100%")

    files1 = listdir(dir1)
    files2 = listdir(dir2)

    files_per_lang = {}

    for f in files1:
        if f == "urls.txt":
            logging.info("skipping (1) 'urls.txt'")
            continue

        lang = f.split(lsep)[fsp1]

        if lang not in files_per_lang:
            files_per_lang[lang] = {"f1": [], "f2": []}

        files_per_lang[lang]["f1"].append(f)

    for f in files2:
        if f == "urls.txt":
            logging.info("skipping (2) 'urls.txt'")
            continue

        lang = f.split(lsep)[fsp2]

        if lang not in files_per_lang:
            files_per_lang[lang] = {"f1": [], "f2": []}

        files_per_lang[lang]["f2"].append(f)

    if opt1 != opt2:
        logging.warning("likely, there will not be identical records since the embeddings are using different optimization strategies")

    for l in files_per_lang:
        len1 = len(files_per_lang[l]['f1'])
        len2 = len(files_per_lang[l]['f2'])
        max_correct = min(len1, len2)
        corrects = 0
        identical = 0
        at_least_one_match_all_embeddings = True

        logging.info(f"detected lang {l}: {len1} (processing {int(len1 * perc1)}) vs {len2} (processing {int(len2 * perc2)})")

        if len1 != len2:
            logging.warning(f"not same quantity of embeddings for lang. {l}")

        files_1 = files_per_lang[l]['f1'][:int(len1 * perc1)]
        files_2 = files_per_lang[l]['f2'][:int(len2 * perc2)]

        for idx, i in enumerate(files_1):
            m = False

            for j in files_2:
                match = compare_embeddings.compare(f"{dir1}/{i}", opt1, f"{dir2}/{j}", opt2, dim=dim)
                ident = compare_embeddings.identical(f"{dir1}/{i}", opt1, f"{dir2}/{j}", opt2, dim=dim)

                if match:
                    corrects += 1
                    m = True
                if ident:
                    identical += 1
                    m = True
                if (m and cont):
                    break

            logging.debug(f"files processed: {idx + 1} of {len(files_1)} ({(idx + 1) * 100.0 / len(files_1):.2f}%)")

            if not m:
                at_least_one_match_all_embeddings = False

        print(f" - {corrects} of {max_correct} detected as correct (at least one match every embedding: {at_least_one_match_all_embeddings})")
        print(f" - {identical} of {max_correct} detected as identical")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility to check the loss of precision when applying some embedding optimization (directory)')

    # Mandatory
    parser.add_argument('embedding_dir_1', metavar='embedding-dir-1',
                        help='Directory 1 of embeddings')
    parser.add_argument('embedding_dir_2', metavar='embedding-dir-2',
                        help='Directory 2 of embeddings')
    parser.add_argument('lang_position_1', metavar='lang-position-1', type=int,
                        help='Since the language is expected to be in the name of the files, separator and position is needed.'
                             ' For instance, if a file is named \'doc_en_2021.txt\', the language is \'en\' and the position is 1')
    parser.add_argument('lang_position_2', metavar='lang-position-2', type=int,
                        help='Since the language is expected to be in the name of the files, separator and position is needed.'
                             ' For instance, if a file is named \'doc_en_2021.txt\', the language is \'en\' and the position is 1')
    parser.add_argument('lang_sep', metavar='lang-sep',
                        help='Since the language is expected to be in the name of the files, separator and position is needed.'
                             ' For instance, if a file is named \'doc_en_2021.txt\', the language is \'en\' and the separator is \'_\'')
    parser.add_argument('embedding_opt_1', metavar='embedding-opt-1', default=None, type=int,
                        help='Embedding optimization which we need to apply in order load the 1st embedding (check embedding_util.py)')
    parser.add_argument('embedding_opt_2', metavar='embedding-opt-2', default=None, type=int,
                        help='Embedding optimization which we need to apply in order load the 2nd embedding (check embedding_util.py)')

    # Other
    parser.add_argument('--cont-when-match', action='store_true',
                        help='When there is a match, do not keep on looking matches for that embedding. This option will speed up the execution')
    parser.add_argument('--percentage-dir-1', type=float, metavar='F', default=1.0,
                        help='Percentage of embeddings to process from directory 1')
    parser.add_argument('--percentage-dir-2', type=float, metavar='F', default=1.0,
                        help='Percentage of embeddings to process from directory 2')
    parser.add_argument('--dim', type=int, metavar='N', default=DEFAULT_EMBEDDING_DIM,
                        help='Embedding dimensionality')
    parser.add_argument('--logging-level', metavar='N', type=int, default=30,
                        help='Logging level. Default value is 30, which is WARNING')

    args = parser.parse_args()

    utils.set_up_logging(level=args.logging_level)

    main(args)
