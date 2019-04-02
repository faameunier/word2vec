from __future__ import division
import os

# The code is optimized for a single threaded BLAS
# Multithreaded BLAS would mess with mp.Pool resulting in
# a major slowdown. Here we force the 2 most commonly
# used BLAS to be single threaded.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VECLIB_NUM_THREADS'] = '1'

import argparse
import pickle
from time import time
from multiprocessing import Pool
import logging
from contextlib import closing
import collections
from random import shuffle, choices, random
from multiprocessing import cpu_count

# useful stuff
import numpy as np
from scipy.special import expit
from scipy.spatial import distance

from utils import *
from pmi import PPMI, Counter
import mp_utils

from nltk.corpus import brown  # used for benchmarking

__authors__ = ['François Meunier', 'Merlin Laffitte', 'Horace Guy']
__emails__ = ['francois.meunier@student.ecp.fr', 'merlin.laffitte@student.ecp.fr', 'horace.guy@student.ecp.fr']


class SkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize=5, minCount=5, sampling_rate=0.001, init="random"):
        self.word_embeddings = None  # place holder
        self.context_embeddings = None  # place holder
        self.n_embed = nEmbed  # size of the embeddings
        self.k = negativeRate  # number of noisy example for each context word
        self.context_size = winSize  # size of the context window
        self.min_count = minCount  # minimum number of times a word must appear to be kept in the vocabulary
        self.word2id = {}
        self.id2word = []
        self.noise_probas = []
        self.sampling_probas = []
        self.vocab_size = 0
        self.sentences = sentences
        self.embeddings = None  # place holder
        self.sampling_rate = sampling_rate
        self.init = init

    @initialized
    def train(self, max_stepsize=0.025, min_stepsize=0.005, epochs=5, chunk_size=100000, n_worker=4):
        """Train the skip-gram model

        Performs parallel training of the skip-gram model
        via asynchronous SGD.

        The stepsize of the sgd decreases over the epochs as
        per Mikolov et al. implementation. The decrease is linear.

        The data is split in chunk_size sub dataset that are given
        to the workers.

        Chunksize directly affects the amount of RAM needed to treat the data, the smaller
        the less RAM is used. However bigger chunks should show some speed improvement.

        Decorators:
            initialized

        Keyword Arguments:
            max_stepsize {number} -- Maximum stepsize of the SGD (default: {0.025})
            min_stepsize {number} -- Minimum stepsize of the SGD (default: {0.0001})
            epochs {number} -- Number of time to iterate over the data (random, some data may not be trained upon) (default: {5})
            chunk_size {number} -- Size of the dataset that each worker has to treat (default: {100000})
            n_worker {number} -- Number of parallel workers (default: {4})
        """
        if self.sentences is None:
            raise NotInitialized("You cannot train a loaded model.")

        logging.info("TRAINING")

        # Preparing shared memory
        # -------------------------------------------
        # Only the embeddings are locked, all the other variables
        # are only read in the code, which speeds up the parallelization
        vocab_ = mp_utils.int2Val(self.vocab_size)
        k_ = mp_utils.int2Val(self.k)
        context_size_ = mp_utils.int2Val(self.context_size)
        noise_probas_ = mp_utils.arr2Arr(self.noise_probas)
        w_emb_ = mp_utils.emb2Arr(self.word_embeddings)
        c_emb_ = mp_utils.emb2Arr(self.context_embeddings)
        all_ids_ = mp_utils.arr2Arr(np.arange(self.vocab_size), True)  # negative sampling speed improvement

        pool_initargs = (vocab_, k_, context_size_, noise_probas_, w_emb_, c_emb_, all_ids_,)

        # You need at least 2 workers, one for preprocessing, one for gradient update
        # -------------------------------------------
        n_worker = max(2, n_worker)

        logging.info("----------------------------------------------")
        logging.info("Initially " + str(self.total_words) + " words")

        # Preprocessing sentences
        # -------------------------------------------
        sentences_ided, n_words = self._subsamble_and_ided_corpus(self.sentences)
        logging.info(str(n_words) + " words kept")

        logging.info("Chunk size: " + str(chunk_size))
        logging.info(str(epochs) + " epochs")
        logging.info(str(n_worker) + " total workers")

        # Guessing the best split between all workers
        # -------------------------------------------
        # This is a little bit overkill on small datasets
        # but very usefull when you start to hit a large
        # vocabulary. In some cases sampling negative
        # examples is slower than computing the gradients.
        # This piece of code has a computational cost at the beginning
        # but saves a lot of time by maximizing the CPU usage.
        TESTSIZE = 10000
        mp_utils.init(*pool_initargs)

        t0 = time()
        temp = []
        for s in sentences_ided[:TESTSIZE]:
            temp.append(SkipGram._build_datasets(s))
        t0 = (time() - t0)

        t1 = time()
        for ds in temp:
            for d in ds:
                if d:
                    SkipGram._parallel_grads((d, 0))
        t1 = (time() - t1)

        n_worker_iterator = max(1, min(int(n_worker / (1 + t1 / t0)), n_worker - 1))
        n_worker_pool = n_worker - n_worker_iterator

        logging.info(str(n_worker_iterator) + " workers on preprocessing")
        logging.info(str(n_worker_pool) + " workers on gradient computation")
        logging.info("----------------------------------------------")

        # Creating the pool of worker for gradient update
        # -------------------------------------------
        p = Pool(n_worker_pool, initializer=mp_utils.init, initargs=pool_initargs)
        for i in range(epochs):
            t = time()
            total_loss = 0

            # shuffling data
            shuffle(sentences_ided)

            logging.info("Epoch " + str(i + 1) + "/" + str(epochs) + " - Data shuffled - " + str(int((time() - t) * 1000) / 1000) + " seconds elapsed")

            # creating the parallel iterator - see mp_utils.py for more details
            iterator = mp_utils.build_iterator(SkipGram._build_datasets, sentences_ided, max_stepsize - i / epochs * (max_stepsize - min_stepsize), n_worker_iterator, pool_initargs)

            # updating embeddings
            for j, loss in enumerate(p.imap_unordered(SkipGram._parallel_grads, iterator, chunksize=chunk_size), 1):
                if j % chunk_size == 0:
                    logging.info("Epoch " + str(i + 1) + "/" + str(epochs) + " -  " + str(int(j / n_words * 100)) + "% - " + str(int((time() - t) * 1000) / 1000) + " seconds elapsed")
                total_loss += loss

            # End of epoch
            logging.info("Epoch " + str(i + 1) + "/" + str(epochs) + " - Total " + str(total_loss) + " - Loss per word " + str(total_loss / n_words) + " - " + str(int((time() - t) * 1000) / 1000) + " seconds elapsed")
        logging.info("----------------------------------------------")

        p.close()
        p.join()

        # updating model embeddings
        self.word_embeddings = mp_utils.Arr2emb(w_emb_, v=self.vocab_size).copy()
        self.context_embeddings = mp_utils.Arr2emb(c_emb_, v=self.vocab_size).copy()

        # cleaning shared memory
        del vocab_, k_, context_size_, noise_probas_, w_emb_, c_emb_, all_ids_

        # setting up final updatings
        self.embeddings = self.word_embeddings + self.context_embeddings

    @initialized
    def save(self, path):
        """Save the current model

        Using pickle.

        Decorators:
            initialized

        Arguments:
            path {str} -- where to save the model
        """
        as_dict = {"word_embeddings": self.word_embeddings,
                   "context_embeddings": self.context_embeddings,
                   "word2id": self.word2id,
                   "id2word": self.id2word,
                   "noise_probas": self.noise_probas,
                   "sampling_probas": self.sampling_probas,
                   "n_embed": self.n_embed,
                   "k": self.k,
                   "context_size": self.context_size,
                   "min_count": self.min_count,
                   "vocab_size": self.vocab_size,
                   "sampling_rate": self.sampling_rate}
        with open(path, 'wb') as f:
            pickle.dump(as_dict, f)
        logging.info("Model succesfully saved:")
        logging.info("----------------------------------------------")
        logging.info("Vocabulary: " + str(self.vocab_size) + " words.")
        logging.info("Embeddings size: " + str(self.n_embed))
        logging.info("Negative sampling factor: " + str(self.k))
        logging.info("Context size: " + str(self.context_size))
        logging.info("----------------------------------------------")
        del as_dict

    @initialized
    def similarity(self, word1, word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        return self._similarity(self.encode(word1), self.encode(word2))

    @staticmethod
    def _similarity(emb1, emb2):
        """Computes similarity between 2 vectors

        Cosine similarity, as per requested shifted
        to output \in [0,1].

        Arguments:
            emb1 {np.ndarray} -- A vector of size n
            emb2 {np.ndarray} -- A vector of size n

        Returns:
            number -- cosine similarity
        """
        try:
            return (2 - distance.cosine(emb1, emb2)) / 2
        except Exception:
            return 0

    @staticmethod
    def load(path):
        """Load a local model

        Loads the locally stored model and sets
        all attributes accordingly. This unlocks almost
        all methods of the object.

        Arguments:
            path {str} -- path to the saved model.
        """
        skip = SkipGram(None)
        with open(path, 'rb') as f:
            skip_dict = pickle.load(f)
            skip.word_embeddings = skip_dict["word_embeddings"]
            skip.context_embeddings = skip_dict["context_embeddings"]
            skip.word2id = skip_dict["word2id"]
            skip.id2word = skip_dict["id2word"]
            skip.noise_probas = skip_dict["noise_probas"]
            skip.sampling_probas = skip_dict["sampling_probas"]
            skip.n_embed = skip_dict["n_embed"]
            skip.k = skip_dict["k"]
            skip.context_size = skip_dict["context_size"]
            skip.min_count = skip_dict["min_count"]
            skip.vocab_size = skip_dict["vocab_size"]
            skip.sampling_rate = skip_dict["sampling_rate"]
            del skip_dict
        logging.info("Model succesfully loaded:")
        logging.info("----------------------------------------------")
        logging.info("Vocabulary: " + str(skip.vocab_size) + " words.")
        logging.info("Embeddings size: " + str(skip.n_embed))
        logging.info("Negative sampling factor: " + str(skip.k))
        logging.info("Context size: " + str(skip.context_size))
        logging.info("----------------------------------------------")
        skip.embeddings = skip.word_embeddings + skip.context_embeddings
        return skip

    @initialized
    def syntactic_guess(self, base_word, role_word, new_word, n=5):
        """Syntactic test

        Given a couple of example words and a new one,
        guesses n words with same role as the role_word
        compared to the base_word regarding the new_word.

        Decorators:
            initialized

        Arguments:
            base_word {str} -- first word in the role example
            role_word {str} -- second word with a specific role regarding the base word
            new_word {str} -- word for which you want a new word with similar role relation as the example ones

        Keyword Arguments:
            n {number} -- number of output words (default: {5})

        Returns:
            list -- predicted words
        """
        base_word_ = self.encode(base_word)
        role_word_ = self.encode(role_word)
        new_word_ = self.encode(new_word)
        temp = self.closest_from_vector(role_word_ - base_word_ + new_word_, n + 3)
        res = []
        for w in temp:
            if w not in [base_word, role_word, new_word]:
                res.append(w)
        return res[:n]

    @staticmethod
    def _build_datasets(sentence_ided):
        """Build datasets

        Given a preprocessed sentence (of ids)
        builds triplets of words, contexts, and class of contexts.

        A lot has been tried to speed up this function.

        This function can only be run in correctly initialized
        parallel context.

        Decorators:
            initialized

        Arguments:
            sentence_ided {list} -- list of ids

        Returns:
            list -- array of triplets
        """
        k = mp_utils.k_factor.value
        context_size = mp_utils.context_size.value
        noise_probas = mp_utils.Arr2arr(mp_utils.noise_probas)
        noise_ids = mp_utils.Arr2arr(mp_utils.all_ids, True)

        # Map should be faster than a for loop
        def speed_me_up(index, w_id):
            temp_context = [c_id for c_id in sentence_ided[max(0, index - context_size):index]] + [c_id for c_id in sentence_ided[index + 1:min(len(sentence_ided), index + context_size + 1)]]  # Should be faster than a for loop
            if temp_context:
                negative_examples = SkipGram.negative_sampling(noise_ids, len(temp_context) * k, noise_probas, [w_id] + temp_context)
                all_examples = np.concatenate((temp_context, negative_examples))
                kron_neg = np.ones_like(all_examples)
                kron_neg[len(temp_context):] = 0
                return [w_id, all_examples, kron_neg]
            return []

        res = [speed_me_up(index, w_id) for index, w_id in enumerate(sentence_ided)]
        return list(filter(lambda x: x is not None, res))

    @initialized
    def _subsamble_and_ided_corpus(self, sentences):
        """Subsamples and converts sentences

        Subsample most common words as per Mikolov and al.
        However there are little informations about when
        to subsample. here we only do it once at the
        beginning of the script.

        Words are also converted to ids.

        Decorators:
            initialized

        Arguments:
            sentences {np.ndarray} -- list of word sentences

        Returns:
            list -- list of id sentences
        """
        logging.info("Subsampling and converting corpus")
        final = []
        final_count = 0
        unk_id = self.id2word[-1]
        for s in sentences:
            temp_s = []
            for word in s:
                w_id = self._word2id(word)
                if w_id != unk_id:
                    if random() > self.sampling_probas[w_id]:
                        temp_s.append(w_id)
                        final_count += 1
            if temp_s:
                final.append(temp_s)
        return final, final_count

    def init_embeddings(self, init=None, corpus=None):
        """Initiate the skipGram model

        Initialisation can be made on a special set
        of sentences if required. If none is given
        it will be based on the ones passed during
        instanciation.

        The initialization is made using a PPMI matrix
        computed by the class pmi.PPMI or randomly.

        Additionnal informations are computed such as the
        probability of noise selection; that is required
        for negative sampling.

        Keyword Arguments:
            init {str} -- init type (default: {None})
            corpus {list} -- list of sentences (default: {None})
        """
        if corpus is None:
            corpus = self.sentences
        if init is None:
            init = self.init
        counter = None
        if init == "ppmi":
            counter = PPMI(corpus, winSize=self.context_size, minCount=self.min_count)
        elif init == "random":
            counter = Counter(corpus, winSize=self.context_size, minCount=self.min_count)
        else:
            raise ValueError("Unkwown init type. Available : random, ppmi")
        self.word2id = counter.word2id
        self.id2word = counter.id2word
        self.noise_probas = counter.unigram_counts ** (3 / 4) / np.sum(counter.unigram_counts ** (3 / 4))
        self.sampling_probas = 1 - np.sqrt(self.sampling_rate / (counter.unigram_counts / np.sum(counter.unigram_counts)))
        self.sampling_probas = np.array([max(0, p) for p in self.sampling_probas])
        self.vocab_size = counter.size
        if init == "ppmi":
            counter.compute_cooccurence()
            counter.compute_ppmi()
            self.word_embeddings = ppmi2embeddings(counter.ppmi, self.n_embed)
            self.context_embeddings = ppmi2embeddings(np.transpose(counter.ppmi), self.n_embed)
        elif init == "random":
            self.word_embeddings = np.random.rand(self.vocab_size, self.n_embed)
            self.context_embeddings = np.random.rand(self.vocab_size, self.n_embed)
        self.total_words = counter.total_count
        del counter

    @initialized
    def encode(self, word):
        """Perform a word2vec conversion

        Returns the word's current embedding.
        If the word is unkown, returns the <UNK>
        embedding.

        Decorators:
            initialized

        Arguments:
            word {str} -- word to encode

        Returns:
            np.ndarray -- the word's embedding
        """
        word = ''.join(e for e in word.lower() if e.isalnum())
        return self.embeddings[self._word2id(word)]

    @initialized
    def closest_from(self, word, n=5):
        """Closest word search

        Find the closest words to a given one.
        The input word will be discarded.

        Decorators:
            initialized

        Arguments:
            word {str} -- input word
            n {int} -- number of similar words to output (default: {5})

        Returns:
            list -- list of similar words
        """
        words = self.closest_from_vector(self.encode(word), n + 1)
        final = []
        for w in words:
            if w != word:
                final.append(w)
        return final[:n]

    @initialized
    def closest_from_vector(self, w_emb, n=5):
        """Closest vector search

        Find the closest word to a given vector

        Decorators:
            initialized

        Arguments:
            w_emb {np.ndarray} -- input vector
            n {int} -- number of similar words to output (default: {5})

        Returns:
            list -- list of similar words
        """
        sim = np.apply_along_axis(lambda emb: self._similarity(emb, w_emb), 1, self.embeddings)
        return [self.id2word[c_id] for c_id in reversed(np.argsort(sim)[-n:])]

    @staticmethod
    def negative_sampling(noise_ids, k, noise_probas, exception_ids):
        """Draw noise words for negative sampling

        Draws k words, using the noise_probas,
        excluding words from the exception_ids.

        Decorators:
            initialized

        Arguments:
            noise_ids {np.ndarray} -- list of word ids, size n
            k {int} -- number of word to draw
            noise_probas {np.ndarray} -- probability of each noise_id
            exception_ids {list} -- list of word to exclude

        Returns:
            np.ndarray -- list of noisy words
        """
        p = noise_probas.copy()
        np.put(p, exception_ids, 0)  # setting the exception probabilities to 0
        p = np.cumsum(noise_probas)  # improves speed of choices, see random_speed_test.py
        return choices(noise_ids, k=k, cum_weights=p)

    @staticmethod
    def _parallel_grads(d):
        """Job to be completed by a worker

        Parallel updates of embeddings.
        The update is performed on the shared memory
        that is initiated by the Pool object.

        Updates are made asynchronously, following
        the standard asynchronous SGD algorithm.

        Arguments:
            d {tuple} -- first element shoud be a dataset triplet and the second one the stepsize

        Returns:
            number -- total loss on the dataset
        """
        word_embeddings = mp_utils.Arr2emb(mp_utils.word_embeddings)  # np.array
        context_embeddings = mp_utils.Arr2emb(mp_utils.context_embeddings)  # np.array
        return SkipGram.compute_grads(d[0], word_embeddings, context_embeddings, d[1])

    @staticmethod
    def compute_grads(dataset, word_embeddings, context_embeddings, eta):
        # OK with 1 word, 1 context et k noise
        # OK with 1 word, 2c context, 2kn noise
        # OK Fully debugged and accuracy tested
        # Compute the gradient of the loss regarding x_words
        # Not sure if we can vectorize it further
        w_id, all_examples, kron_neg = dataset
        w = word_embeddings[w_id]  # 1 x N_emb
        n_context = collections.Counter(kron_neg)[1]
        M_contexts = context_embeddings[all_examples]  # 2C(K+1) x N_emb
        similarities = np.dot(M_contexts, w)  # 2C(K+1) x 1 (transposed)
        probas = expit(similarities)  # 2C(K+1) x 1 (transposed)
        helper = probas - kron_neg  # 1 x 2C(K+1)
        context_embeddings[all_examples] += -eta * np.outer(helper, w)  # This is actually wrong. If a context word appears several times it would only be updated once. However, the speed up is x3
        # np.add.at(context_embeddings, all_examples, -eta * np.outer(helper, w))  # update context embeddings multiple time if necessary
        word_embeddings[w_id] += -eta * np.dot(helper, M_contexts)
        # matmul_addat(context_embeddings, all_examples, -eta * np.outer(helper, w))
        # OLD VERSION - less vectorized
        # Mc = context_embeddings[y_contexts[i]]
        # Mn = context_embeddings[y_noise[i]]
        # pi = expit(np.dot(Mc, w))
        # qk = expit(-np.dot(Mn, w))
        # loss += -np.sum(np.log(pi)) - np.sum(np.log(qk))
        # word_embeddings[w_id] -= eta * (np.sum((pi - 1)[:, np.newaxis] * Mc, axis=0) + np.sum((1 - qk)[:, np.newaxis] * Mn, axis=0))  # utiliser np.outer
        # np.add.at(context_embeddings, y_contexts[i], -eta * (pi - 1)[:, np.newaxis] * np.array([w] * Mc.shape[0]))  # could be faster # utiliser np.outer
        # np.add.at(context_embeddings, y_noise[i], -eta * (1 - qk)[:, np.newaxis] * np.array([w] * Mn.shape[0]))  # could be faster # utiliser np.outer
        return -np.sum(np.log(expit(-similarities[n_context:]))) - np.sum(np.log(expit(similarities[:n_context])))

    @initialized
    def _word2id(self, word):
        """Word to Id

        Convert a word in an Id.
        If unknown, returns the <UNK> token ID

        Arguments:
            word {str} -- word

        Returns:
            int -- ID of the word if kwnown
        """
        try:
            return self.word2id[word]
        except KeyError:
            return len(self.word2id) - 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')
    parser.add_argument('--debug', help='enters debug mode', action='store_true')

    opts = parser.parse_args()

    n_cpu = cpu_count()

    if opts.debug:
        logging.basicConfig(filename='debug.log', level=logging.DEBUG)
        logging.debug("===DEBUG/TEST MODE===")
        sentences = preprocess_sentences(brown.sents())
        #  sentences = text2sentences(opts.text)
        sg = SkipGram(sentences, nEmbed=100, negativeRate=5, winSize=5, minCount=5)
        sg.init_embeddings()
        # gradient_checker(sg.compute_f_helper, np.concatenate((sg.word_embeddings, sg.context_embeddings)))
        sg.train(epochs=3, n_worker=n_cpu)
        logging.debug(sg.closest_from("saturday", 10))
        logging.debug(sg.closest_from("money", 10))
        logging.debug(sg.closest_from("child", 10))
        sg.save(opts.model)
        sg = SkipGram.load(opts.model)
        logging.debug(sg.closest_from("saturday", 10))
        logging.debug(sg.closest_from("money", 10))
        logging.debug(sg.closest_from("child", 10))

    else:
        logging.basicConfig(filename='info.log', level=logging.INFO)
        if not opts.test:
            sentences = text2sentences(opts.text)
            sg = SkipGram(sentences, nEmbed=200, negativeRate=5, winSize=5, minCount=5)
            sg.init_embeddings()
            sg.train(epochs=3, n_worker=n_cpu)
            sg.save(opts.model)

        else:
            pairs = loadPairs(opts.text)
            sg = SkipGram.load(opts.model)
            for a, b, _ in pairs:
                print(sg.similarity(a, b))
