# from skipGram import SkipGram
import utils
from nltk.corpus import brown
import logging
from gensim.models.word2vec import Word2Vec
# from pmi import Counter


logging.basicConfig(filename='gensim_debug.log', level=logging.DEBUG)
sentences = utils.preprocess_sentences(brown.sents())
# sentences = utils.text2sentences("traindata/training-monolingual.tokenized.shuffled/news.en-00001-of-00100")
# c = Counter(sentences)
# for k in c.word2id.keys():
#     assert c.id2word[c.word2id[k]] == k
#     c.unigram_counts[c.word2id[k]]
model = Word2Vec(sentences, size=200, window=5, min_count=5, workers=4, negative=15, iter=5, compute_loss=True)
model.save("gensim_sg.model")
# model = Word2Vec.load("gensim_sg.model")
# logging.debug(model.accuracy())
logging.debug(model.get_latest_training_loss())
logging.debug(model.most_similar(positive="saturday", topn=10))
logging.debug(model.most_similar(positive="money", topn=10))
logging.debug(model.most_similar(positive="child", topn=10))
