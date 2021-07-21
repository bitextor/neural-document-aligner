
# Neural Document Aligner

![License](https://img.shields.io/badge/License-GPLv3-blue.svg)

`neural-document-aligner` is a tool in Python which uses neural tools in order to align documents from a pair of languages. All what you need to run this tool is provide a set of source documents and another set of target documents with their language code (you can obtain these documents from [OPUS](https://opus.nlpl.eu/)). This tool has been created with the aim of aligning documents crawled from the web, so documents should be related to URLs, but this is optional.

Besides, the tool has been implemented to be ready to be as modular as possible in order to add other strategies for alignment, matching, optimizations, results, etc.

## Installation

To install `neural-document-aligner`, first clone the repository:

```bash
git clone https://github.com/bitextor/neural-document-aligner.git
```

Once the repository has been cloned, run:

```bash
cd neural-document-aligner

# create virtual environment & activate
python3 -m venv /path/to/virtual/environment
source /path/to/virtual/environment/bin/activate

# install dependencies in virtual enviroment
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

### Dependencies

In order to run the neural document aligner we will need to have installed [LASER](https://github.com/facebookresearch/LASER).

As it is explained in the [installation of LASER](https://github.com/facebookresearch/LASER#installation), the environment variable `LASER` is expected to be defined, and we also need this configuration. Run:

```bash
export LASER="/path/to/LASER"

# Optionally
echo "export LASER='/path/to/LASER'" >> ~/.bashrc
```

## Usage

You can easily check the different options running:

```bash
python3 neural_document_aligner/neural_document_aligner.py --help
```

The different parameters are:

```bash
usage: neural_document_aligner.py [-h]
                                  [--docalign-strategy {faiss,lev,lev-full,just-merge}]
                                  [--weights-strategy {0,1,2,3}]
                                  [--merging-strategy {0,1,2,3,4,5}]
                                  [--results-strategy N]
                                  [--gen-emb-optimization-strategy {0,1,2}]
                                  [--emb-optimization-strategy {0,1,2}]
                                  [--processes N] [--workers N] [--dim N]
                                  [--generate-embeddings]
                                  [--generate-and-finish]
                                  [--embed-script-path PATH]
                                  [--random-mask-value <v_1>,<v_2>,...,<v_dim>]
                                  [--check-zeros-mask] [--min-sanity-check N]
                                  [--input-src-and-trg-splitted]
                                  [--do-not-merge-on-preprocessing]
                                  [--gold-standard PATH] [--apply-heuristics]
                                  [--output-with-urls]
                                  [--process-max-entries N]
                                  [--faiss-threshold F]
                                  [--faiss-reverse-direction]
                                  [--faiss-take-knn N] [--logging-level N]
                                  [--log-file PATH]
                                  input-file src-lang trg-lang
```

### Input format

The input file is expected to be TSV (Tab-Separated Values) file. The columns which we expect are:

1. Path to document in language `src-lang`.
2. Path to document in language `trg-lang`.
3. Path to embedding of the document provided in the 1st column.
4. Path to embedding of the document provided in the 2nd column.
5. URL related to the document of the 1st column (other information which is related to the document uniquely can also be provided instead of the URL). '-' if you do not want to provide this information.
6. URL related to the document of the 2nd column (other information which is related to the document uniquely can also be provided instead of the URL). '-' if you do not want to provide this information.

Usually, we have documents with their language code, but we do not have a relation between the documents. In this case, you can use `--input-src-and-trg-splitted` in order to use the following format in the input file:

1. Path to document
2. Path to embedding of the document provided in the 1st column.
3. URL related to the document of the 1st column (other information which is related to the document uniquely can also be provided instead of the URL). '-' if you do not want to provide this information.
4. 'src' if the language of the document provided in the 1st column is `src-lang`. 'trg' if the language of the document provided in the 1st column is `trg-lang`.

### Strategies

Different strategies have been implemented for different actions.

#### Weigh embeddings

As a step of the preprocessing, different strategies can be applied to weight the embeddings. The available strategies are:

1. Do not apply any weight strategy. This strategy can be applied with `--weights-strategy 0`.
2. Weight using the sentence length and the frecuency of the sentence (based on TF), a.k.a. SL. This strategy can be applied with `--weights-strategy 1`.
3. Weight using IDF. This strategy can be applied with `--weights-strategy 2`.
4. Weight using SLIDF. This strategy can be applied with `--weights-strategy 3`.

#### Merge embeddings

As another step of the preprocessing, different strategies can be applied to merge the sentence-level embedding obtained from LASER and get a document-level embedding. Sometimes, the document alignment strategy will apply its own merging strategy because needs information which we do not have in the processing step, and in this case, we can use `--do-not-merge-on-preprocessing` (anyway, the merging strategy may reach, optionally, to the document alignment strategy, and different merging strategies may be applied, just after, not at the preprocessing step). The available strategies at the preprocessing step are:

1. Do not apply any merging strategy. This strategy can be applied with `--merging-strategy 0`.
2. Merge embeddings using the average value of the components. This strategy can be applied with `--merging-strategy 1`.
3. Merge embeddings using the median value of the components. This strategy can be applied with `--merging-strategy 2`.
4. Merge embeddings using the max value of the components. This strategy can be applied with `--merging-strategy 3`.
5. Merge embeddings using the max value of the components of a third, resulting in 3 new embeddings, and apply again the max value of the components. This is an attempt of preserve the order of the sentences. This strategy can be applied with `--merging-strategy 4`.
6. Merge embeddings using the average value of the components, but the average value is recalculated embedding after embedding in a iterative way. This is an attempt of preserve the order of the sentences. This strategy can be applied with `--merging-strategy 5`.

#### Document alignment

Different strategies can be applied to search the matches of the documents. The available strategies are:

* `faiss`: use [Faiss](https://github.com/facebookresearch/faiss) in order to avoid to compare all the document in `src-lang` with all the documents in `trg-lang`. It is the recommended option.
* `lev-full`: use the levenshtein algorithm to compare documents. Every sentence-level embedding of one document is compared with the other and the result of levenshtein algorithm is the score. It is very time-consuming.
* `lev`: use the levenshtein algorithm with optimizations. The main optimizations are using a 2-row matrix instead of a full matrix and only process the central band of the matrix instead of processing all the values.
* `just-merge`: do not use any strategy, just merge the embeddings with a specific strategy and obtain the score based on the cosine distance.

#### Embedding space optimizations

Different strategies can be applied to save up storage usage when the embeddings are generated (the same strategy will be needed to be applied when loading the embeddings). The available strategies are:

1. Do noy apply any optimization. This strategy can be applied with `--emb-optimization-strategy 0` (if you are going to generate embeddings, you will also need `--gen-emb-optimization-strategy 0`).
2. Optimize using `float16` instead of `float32`, what it will decrease the storage usage to the half. This strategy can be applied with `--emb-optimization-strategy 1` (if you are going to generate embeddings, you will also need `--gen-emb-optimization-strategy 1`).
3. Optimize using vector quantization of 8 bits, what it will decrease the storage usage to a quarter. This strategy can be applied with `--emb-optimization-strategy 2` (if you are going to generate embeddings, you will also need `--gen-emb-optimization-strategy 2`).

#### Results

The strategies applied in order to obtain the matches (i.e. results) differ in relation to the selected document alignment strategy. The available strategies are:

* `faiss`:
   1. Search for every document in `src-lang` the `--take-knn` closest documents in `trg-lang`, then sort by distance and once a pair has been selected, will not be selected again. This strategy can be applied with `--results-strategy 0`.
* `lev-full`, `lev` and `just-merge`:
   1. Union of the best matches in both directions (i.e. from `src-lang` to `trg-lang` and from `trg-lang` to `src-lang`). This strategy can be applied with `--results-strategy 0`.
   2. Intersection of the best matches (i.e. the best match for a document from `src-lang` to `trg-lang` is the same in the oposite direction). This strategy can be applied with `--results-strategy 1`.

### Parameters

There are different parameters in order to achieve different behaviours:

* Mandatory:
  * `input-file`: the input file where we will have the necessary information to look for matches of the provided documents.
  * `src-lang`: language code (e.g. 'en', 'fr') of the source documents.
  * `trg-lang`: language code (e.g. 'en', 'fr') of the target documents.
* Optional:
  * Meta:
    * `-h, --help`: help message and exit
  * Strategies:
    * `--docalign-strategy {faiss,lev,lev-full,just-merge}`: strategy to select the matches among the documents.
    * `--weights-strategy {0,1,2,3}`: strategy to weight the embeddings.
    * `--merging-strategy {0,1,2,3,4,5}`: strategy to merge the embeddings.
    * `--results-strategy N`: strategy to handle the results from the document alignment.
    * `--gen-emb-optimization-strategy {0,1,2}`: embedding optimization strategy when the embeddings are being generated (i.e. the embeddings will be stored using the selected strategy).
    * `--emb-optimization-strategy {0,1,2}`: embedding optimization strategy when the embeddings are being loaded.
  * Multiprocessing:
    * `--processes N`: number of processes to use if multiprocessing is possible (is possible in the docalign strategies `lev`, `lev-full` and `just-merge`).
    * `--workers N`: number of workers to use if multiprocessing is possible (is possible in the docalign strategies `lev`, `lev-full` and `just-merge`).
  * Embeddings:
    * `--dim N`: dimensionality of the embeddings. This value might change if you change your embeddings generation source or you apply different processes (e.g. embedding reduction).
    * `--generate-embeddings`: by default, the provided paths to the embeddings in the `input-file` are expected to exist. In the case that you want to generate the embeddings, this option must be set.
    * `--generate-and-finish`: if you just want to generate the embeddings and do not perform the matching of the documents, this option must be set to finish the execution once the embeddings have been generated.
    * `--embed-script-path PATH`: path to the file `embed.py` of [LASER](https://github.com/facebookresearch/LASER). This parameter allows to change the default path, which is the one defined by the LASER installation. The default value is the file `embed.py` which is inside this repository (there are options which are not in the original LASER and are mandatory in order to generate correctly the embeddings). The reason about chanching this file is add different options to optimize the behaviour of LASER without necessary modify your LASER installation.
    * `--random-mask-value <v_1>,<v_2>,...,<v_dim>`: mask to apply to every embedding. If you have a mask which you know that it works better for a specific pair of languagues, you may apply it through this parameter. The expected values are float values.
    * `--check-zeros-mask`: if you want to remove the components of the embeddings which, after applying the mask, the result is 0, this option must be set.
  * Other:
    * `--min-sanity-check N`: number of entries of the `input-file` which will be checked out to be ok.
    * `--input-src-and-trg-splitted`: if you want to provide an `input-file` with the second format instead the default, this option must be set.
    * `--do-not-merge-on-preprocessing`: the merging strategy is applied in a preprocessing step, but some strategies might need to apply the merging strategy by their own (e.g. `lev`). In that case, this option must be set.
    * `--gold-standard PATH`: if you want to obtain the recall and precision of the resulted matches, you need to provide a gold standar with the format 'src_document_path\ttrg_document_path'.
    * `--apply-heuristics`: you can enable heuristics if you set this option. The heuristics are different conditions which makes us to be sure that two documents are not a match even if they have been matched, and with the heuristics that match will be removed.
    * `--output-with-urls`: if you provided URLs in the `input-file` and you want to show them in the results instead of the paths, this option must be set. If this option is set, `--gold-standard PATH` will be expected to contain URLs instead of the paths to the documents.
    * `--process-max-entries N`: max. number of entries to process from the `input-file`.
  * Other (`faiss` docalign strategy):
    * `--faiss-threshold F`: if the score of a match does not reach the provided threshold, it will be discarded.
    * `--faiss-reverse-direction`: instead of going from `src-lang` to `trg-lang` we go from `trg-lang` to `src-lang`.
    * `--faiss-take-knn N`: number of documents to check the distance from one document of `src-lang` to `trg-lang`.
  * Other (logging):
    * `--logging-level N`: level of logging to show the different lines related to a concrete severity. The more verbose value is a value of 0, but the default value is to only show the necessary information (warning and above).
    * `--log-file PATH`: log file where all the logging entries will be stored. Even if this option is set, the logging entries will be showed up in the standar error output (if you want to remove those lines, you will need to redirect the standar error output).

## Example

Example of execution where we do not provide a file but we pipe it. The embeddings do not exist, so we generate them. Different strategies are applied. Evaluation will not be carried out since has not been provided a gold standard file, but we want the output with URLs instead of the paths to the documents.
```bash
# Pipe the file, generate the embeddings and do not apply evaluation
echo -e \
"/path/to/doc1\t/path/to/embedding1\thttps://www.this_is_a_url.com/resource1\tsrc\n"\
"/path/to/doc2\t/path/to/embedding2\thttps://www.this_is_a_url.com/resource2\ttrg"  | \
\
python3 neural_document_aligner/neural_document_aligner.py - en fr \
                                                           --docalign-strategy 'faiss' --weights-strategy 0 \
                                                           --merging-strategy 3 --results-strategy 0 \
                                                           --generate-embeddings --gen-emb-optimization-strategy 2 \
                                                           --emb-optimization-strategy 2 --input-src-and-trg-splitted \
                                                           --output-with-urls --faiss-threshold 0.7
```
