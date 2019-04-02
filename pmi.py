from utils import *
import numpy as np
import logging


class Counter:
    """A simple counter to ease Word2Vec init

    Class performing several usefull counting
    and computation on the corpus, among which:
        - unigram count
        - total unigram count
        - filtering out rare words
        - word2id and reversed

    It is not possible right now to update the data
    after instanciation. This could be added in the future.
    """

    def __init__(self, sentences, winSize=5, minCount=5):
        self.sentences = sentences
        self.min_count = minCount
        self.id2word = []
        self.size = 0  # size of the vocabulary
        self.word2id = {}
        self.context_size = winSize  # size of the context window
        self.total_count = 0  # total number of words in thr corpus
        self.unigram_counts = []  # unigram count
        self.build_vocabulary()

    def build_vocabulary(self):
        """Initialization helper

        Do all the job !
        """
        count = {}
        for s in self.sentences:
            for word in s:
                count[word] = count.get(word, 0) + 1
        delete_me = []
        for w in count.keys():
            if count[w] < self.min_count:
                delete_me.append(w)
        count["<UNK>"] = 0
        for w in delete_me:
            temp = count.pop(w)
            count["<UNK>"] += temp

        vocab = list(count.keys())
        self.id2word = np.array(vocab)

        self.size = len(vocab)
        self.word2id = {word: i[0] for i, word in np.ndenumerate(self.id2word)}

        self.unigram_counts = []
        for w in self.id2word:
            self.unigram_counts.append(count[w])
        self.unigram_counts = np.array(self.unigram_counts)

        self.total_count = sum(self.unigram_counts)
        logging.info("Vocab size: " + str(self.size))
        logging.info("Vocabulary constructed")


class PPMI(Counter):
    """Positive pointwise mutual information

    Extends the Counter class with some additional
    function to compute a ppmi matrix that could be used
    to initialize embedddings.

    Deprecated ?
    """

    def __init__(self, sentences, winSize=5, minCount=5):
        super().__init__(self, sentences, winSize, minCount)
        self.cooccurence = [[]]  # place holder
        self.unigram_counts = []  # unigram count

    def build_vocabulary(self):
        """Initialization helper

        Set sizes of remaining key attributes
        """
        super().build_vocabulary()
        self.cooccurence = np.zeros((self.size, self.size))
        self.ppmi = np.zeros((self.size, self.size))

    def compute_cooccurence(self):
        """Compute cooccurrence matrix."""
        logging.info("Filling cooccurence matrix")
        for s in self.sentences:
            for index, word in enumerate(s):
                # logging.debug("Word:" + word)
                word_id = len(self.id2word)
                if word in self.word2id.keys():
                    word_id = self.word2id[word]
                    for context in s[max(0, index - self.context_size):min(len(s), index + self.context_size + 1)]:
                        # logging.debug("Context:" + context)
                        context_id = len(self.id2word)
                        if context in self.word2id.keys():
                            context_id = self.word2id[context]
                            if context != word:
                                self.cooccurence[word_id, context_id] += 1
        logging.info("Cooccurence matrix computed")

    def compute_ppmi(self):
        """Compute ppmi matrix

        Follows definition: https://web.stanford.edu/~jurafsky/slp3/6.pdf equation (6.20)
        """
        logging.info("PPMI computation")
        total = np.sum(self.cooccurence)
        pi_ = np.apply_along_axis(np.sum, 1, self.cooccurence)
        p_j = np.apply_along_axis(np.sum, 0, self.cooccurence)
        for i in range(len(pi_)):
            for j in range(len(p_j)):
                temp = self.cooccurence[i, j]
                if temp != 0:
                    self.ppmi[i, j] = max(0, np.log2(total * temp / (pi_[i] * p_j[j])))
        logging.info("PPMI done")


if __name__ == '__main__':
    # Some tests
    logging.basicConfig(filename='debug.log', level=logging.DEBUG)
    sentences = text2sentences("some_test_sentences.txt")
    logging.debug(sentences)
    p = PPMI(sentences)

    logging.debug("should output < 57")
    logging.debug(p.size)
    p.run_counts()

    logging.debug("should output 57")
    logging.debug(p.total_count)  # OK
    logging.debug("occurence matrix")
    logging.debug(p.cooccurence)
    p.compute_ppmi()
    logging.debug("ppmi matrix")
    logging.debug(p.ppmi)

    pd.DataFrame(p.cooccurence, columns=p.id2word, index=p.id2word).to_csv("debut_test_ppmi.csv", sep=";")  # OK

    p.cooccurence = np.matrix([[0, 0, 1, 0, 1], [0, 0, 1, 0, 1], [2, 1, 0, 1, 0], [1, 6, 0, 4, 0]])
    p.ppmi = np.zeros((4, 5))
    p.compute_ppmi()
    logging.debug("compare against ref https://web.stanford.edu/~jurafsky/slp3/6.pdf")
    logging.debug(p.ppmi)  # OK
