
import os
import sys
import random
import string
import logging
import argparse

from sentence_splitter import SentenceSplitter, SentenceSplitterException

import utils.utils as utils
from exceptions import FileFoundError

def split(doc_path, lang, output=None, text=None):
    if doc_path is not None:
        doc_path = utils.expand_and_real_path_and_exists(doc_path)

    if (output is not None and output != "-"):
        output = utils.expand_and_real_path_and_exists(output)

        if os.path.exists(output):
            logging.error(f"file '{output}' exists\n")

            return 3, None

    try:
        splitter = SentenceSplitter(language=lang)
    except (SentenceSplitterException, Exception) as e:
        logging.warning(f"{str(e)} (using 'en' as language)")

        splitter = SentenceSplitter(language='en')

    segmented_text = ""

    try:
        if text is None:
            text = ""

            if doc_path == "-":
                text = sys.stdin.read()
            else:
                with open(doc_path, "r") as file:
                    for line in file:
                        text += line

                    text = text.strip()

        segments = splitter.split(text)
        segmented_text = "\n".join(segments) + "\n"

        if output == "-":
            sys.stdout.write(segmented_text)
        elif output is not None:
            with open(output, "w") as file:
                file.write(segmented_text)

    except (SentenceSplitterException, Exception) as e:
        logging.error(f"{str(e)} (writing file to output without splitting)")

        status = 0

        if (output is not None and output != "-"):
            status = os.system(f"cp {doc_path} {output}")
        elif output is not None:
            status = os.system(f"cat {doc_path}")

        if status != 0:
            logging.error(f"could not write the file to output (status of executed command: {status})")

            return 2, text if text is not None else ""
        return 1, text if text is not None else ""
    return 0, segmented_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split document')

    parser.add_argument('input_file', metavar='input-file',
                        help='Input file to split. If \'-\' is provided, the doc will expected to be provided by stdin')
    parser.add_argument('lang',
                        help='Language of the provided input file')

    parser.add_argument('--output', default="-",
                        help='Output file for the results. The default value is \'-\', which means stdout')
    parser.add_argument('--logging-level', metavar='N', type=int, default=30,
                        help='Logging level. Default value is 30, which is WARNING')

    args = parser.parse_args()

    utils.set_up_logging(level=args.logging_level)

    doc_path = args.input_file
    lang = args.lang
    output = args.output

    status, _ = split(doc_path, lang, output=output)

    sys.exit(status)
