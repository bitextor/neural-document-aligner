
import logging

DEFAULT_EMBEDDING_DIM = 768
DEFAULT_ST_MODEL = "LaBSE"
DEFAULT_LOGGING_LEVEL = logging.INFO
DEFAULT_LOGGING_FORMAT = "[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] %(message)s"
DEFAULT_MAX_MBYTES_PER_BATCH = 200 # Change to None or neg. number in order to disabe size limit per batch
DEFAULT_MAX_NOLINES_PER_BATCH = 200000 # Change to None or neg. number in order to disable nolines limit per batch
DEFAULT_BATCH_SIZE = 32
DEFAULT_SENTENCE_SPLITTING = False
DEFAULT_MAX_SENTENCES_LENGTH = 10000 # Change to None for unlimited length

ST_SHOW_PROGRESS = False
