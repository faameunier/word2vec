import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import numpy as np
import logging


def text2sentences(path):
    """Loads a file into sentences

    Read the path and perform some preprocessing
    to generate a list of sentences in lower case and without
    special characters.

    Arguments:
        path {str} -- path towards the file to read

    Returns:
        list -- list of sentences
    """
    sentences = []
    with open(path, encoding="utf8") as f:
        for l in f:
            sentences.append(l.split())
    sentences = preprocess_sentences(sentences)
    return np.array(sentences)


def preprocess_sentences(corpus):
    """Preprocess sentences

    lower words and only keeps
    alnum characters. Empty words
    and sentences are removed from the corpus

    Arguments:
        corpus {list} -- list of list of words

    Returns:
        np.ndarray -- Preprocessed sentences
    """
    sentences = []
    for sentence in corpus:
        s = []
        for word in sentence:
            treated = ''.join(e for e in word.lower() if e.isalnum())
            if treated != '':
                s.append(treated)
        if len(s) > 0:
            sentences.append(s)
    return np.array(sentences)


def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'], data['word2'], data['similarity'])
    return pairs


class NotInitialized(Exception):
    """An exception that is raised if the embeddings were not initialized"""

    def __init__(self, msg="Embeddings are not initialized, please run init_embeddings or load an existing model"):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)


def initialized(func):
    """Initialized wrapper

    Checks that the skipGram embeddings are initialized

    Returns:
        Unknown -- fun output

    Raises:
        NotInitialized -- The module is not initialized correctly
    """
    def check_auth(*args, **kwargs):
        skip = args[0]
        if skip.word_embeddings is None or skip.context_embeddings is None:
            raise NotInitialized()
        return func(*args, **kwargs)
    return check_auth


def ppmi2embeddings(ppmi, n_emb):
    """Convert PPMI matrix to embeddings

    Do a simple conversion by performing
    a normalization of the matrix and
    dimensionnal reduction via PCA
    or augmentation by padding the data.

    Arguments:
        ppmi {np.ndarray} -- PPMI matrix
        n_emb {int} -- size of the embeddings

    Returns:
        np.ndarray -- embeddings matrix
    """
    shape = ppmi.shape
    temp = ppmi
    logging.debug(shape)
    logging.debug(temp)
    if shape[1] > n_emb:
        logging.info("PPMI reduction via PCA")
        temp = PCA(n_components=n_emb).fit_transform(temp)
        # logging.debug(temp)
        # logging.debug((temp.shape))
        # logging.debug(type(temp))
        return normalize(temp) - 0.5
    elif shape[1] < n_emb:
        logging.info("PPMI augmentation via padding")
        temp = np.pad(temp, ((0, 0), (0, n_emb - shape[1])), mode='constant', constant_values=(0, 0))
        # logging.debug(temp)
        # logging.debug((temp.shape))
        # logging.debug(type(temp))
        return normalize(temp) - 0.5
    else:
        logging.info("PPMI normalization")
        # logging.debug(type(temp))
        return normalize(temp) - 0.5
