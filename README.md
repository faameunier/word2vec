# Motivation
This project was conducted as a homework for the course **Natural Language Processing E4-3** teached by Matthias Galle. The subject was to reimplement the SkipGram with negative sampling model, developped by Mikolov and al., from scratch.
Special care was taken to optimize the code as much as possible. **The Brown corpus was used to benchmark the performance of the code with a size of embeddings of 200, a context windows size of 5, and 5 negative example for each context word**.
This ReadMe will go over the basic ideas and problems faced, but will not deep dive into the mathematical model. For such explanation please review the bibliography, and specifically [2], [3] and [5]. We will present the problems in a somewhat  chronological order.

# Toward a first solution...
What follows are the key points of the first implemented algorithm.

### Corpus preprocessing
The vocabulary is built by reading a first time all sentences. Each words are lowered and any special character is removed from the dataset. A <UNK> token is added to represent any unkwnown word.

### Model loading and saving
Model are saved and loaded using **pickle**. A nice speed improvement could be used by replacing the library by **cPickle**, but we wanted to keep the requirement list as small as possible. Anyways, even the biggest model trained (with a vocabulary of 70k words) wasn't bigger than a few hundreds MB and was still fast enough to save and load on our test setup.

The saved model can be trained again, given that you manually set additional sentences to the loaded SkipGram object. 
### Negative sampling
Negative sampling was one of the first method we implemented. We used the same sampling probability formula as in [3]. We encountered several problems especially regarding speed during implementation.

The first solution was to leverage on **numpy.random.choice**. It brought up two problems:
- Difficulty to parallelize the random state, which required some overhead that we didn't want (see https://stackoverflow.com/questions/29854398/seeding-random-number-generators-in-parallel-programs for instance)
- Slowness of the solution as we are only selecting a very limited number of words with replacement out of a huge vocabulary. This is a known problem (see https://github.com/numpy/numpy/issues/2764).

The solution implemented was to leverage on Python 3.6 brand new **random.choices** method that is basically a clone of numpy's method. However it is implemented in **C**, and can directly accept cumulative probability sum for a free extra speed up. Some timeit tests are provided in **random_speed_test.py**.

### Mathematical formulas and vectorizing the model
The model was implemented as per [5], more precisely equations (59) and (61). It is worth noting that for a given word, all computation and updates of the output layer can be made in a vectorized manner. The initial implementation used to split context words and negative samples. Later on we noticed that providing a vector of 1s and 0s depending on the class of the context word (real context or negatively sampled) would allow further vectorization. This vector is referred to as **kron_neg** in the code. Vectorizing the operation had a big impact on speed.

However we couldn't find any ways to compute the gradient by batch of words instead of performing a **for-loop**.

The stepsize of the **stochastic gradient descent** was made similarly as Mikolov and al. original code (linearly decreasing over epochs).

### Model initialization
Two strategy were tested:
- PPMI matrix: our first hope was to compute a PPMI matrix (see [1] for details) as initial embeddings for words, and transpose it for the contexts. To set the matrices to expected sized a principal component analysis was to be used (or padding if required). This solution seemed to work fine on small datasets but it is quickly not viable on big datasets. Even when using sparse matrices (deprecated code), the computational time was too long to be reasonnable.
- Random state: the simpler the better, we finally used random matrices for embeddings and got very similar (if not equal) results compared to PPMI. This solution is implemented right now and any code regarding PPMI should be considered as deprecated (no guarantee that the code actually works)

# The first model was a huge failure. Step two !
Our first model had 3 *major* problems:
- Poor speed, with less than 400 words/sec (vectorizing the model got us to about 1k words/sec)
- High RAM impact (5Gb to treat the first 100k lines of Google's Billion word corpus)
- Poor results: when training on the Brown corpus and asking for 5 similar words to "saturday", we got what seemed like random words. Whereas **gensim** Word2Vec's implementation was giving almost perfectly the full week with similar hyperparameters (let's not talk about training speed as it would be depressing - gensim is implemented in **Cython**)
### Why the model didn't work ?
Diving back into Mikolov and al. paper [2], we noticed the only apparent discrepancy between their algorithm and ours is **subsampling**. In the original paper it is stated:
> In very large corpora, the most frequent words can easily occur hundreds of millions of times (e.g.,
“in”, “the”, and “a”). Such words usually provide less information value than the rare words.

At first you could misinterprete 'very large corpora' as a billion of words. Truth is, even on small datasets, subsampling is key. There are little information on how to subsample the corpus. We implemented the subsampling as a preprocessing pass. While the words are converted into ids, too frequent words are removed definitely from the sentences. This is done in **SkipGram._subsamble_and_ided_corpus()** and follows once again the original implementation as per [2].

The model finally started to give decent similar words and similarity measure. Two problems to go.

### Low memory usage
The high memory usage was mostly due to the fact the we preprocessed the whole corpus and stored it in RAM. This could be avoided thanks to **iterators**. However, this proved slightly more technical to implement than expected when it came to speed...

### Need for speed
Let's face it, as a pure python implementation (numpy put aside), this code is not going to be the faster Word2Vec implementation ever. However we can still benchmark against a very interesting blog post from gensim's founder Radim Řehůřek : https://rare-technologies.com/word2vec-in-python-part-two-optimizing/ and the following article about multiprocessing (in **C**).

First of all we changed the update formula slightly. In fact, theorically a word can come up several time as a context of a given word. This is a problem. Indeed, numpy array "+=" operation is unable to update several times the same line when used with a mask. Our initial implementation used **numpy.add.at()** function to update the embeddings in a mathematically correct way. In our final code we swapped back to the good ole' "+=" operation as it is vectorized in **C** and was 2 times faster on our benchmark. This should have little to no impact on the final result : context embeddings are likely to converge slightly less fast, but this is a risk we are willing to take.

We parallelized the SGD following an asynchronous SGD algorithm (see [6]). Extra care was taken to only lock the embeddings (as they are the only writteable data). This allowed us to reach **8k words/sec** (when batching the data in large enough chunks). However we were facing starvation of our workers : our data iterator was to slow to feed the workers ! That was unexpected.

Next step was to parallelize the preprocessor iterator (which is a nonsense and impossible in general). However as the preprocessor actually iterates on sentences, a solution is to parallelize the treatment of the sentences and stop the execution of the pool each time a worker is done (see **mp_utils.py**). 

Thanks to all these steps, we reached **13k words/sec**. A final step was to automatically estimate how many workers should be used to work on the preprocessing part, and how many on linear algebra operations (regarding this, you should use a monothreaded BLAS to run this code. We try to enforce it in the code but there are no guarantees that will work on all hardwares and compiled versions of numpy).

# Resources
- [1] Daniel J. & James H. M. "Vector semantics",  *Speech and Language Processing.* 2008, retrieved from https://web.stanford.edu/~jurafsky/slp3/6.pdf
- [2] Mikolov, T., Sutskever, I., Chen, K., Corrado, G.S., & Dean, J. (2013). Distributed Representations of Words and Phrases and their Compositionality. *NIPS.*
- [3] Mikolov, T., Chen, K., Corrado, G.S., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. *CoRR, abs/1301.3781.*
- [4] Goldberg, Y., & Levy, O. (2014). word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method. *CoRR, abs/1402.3722.*
- [5] Rong, X. (2014). word2vec Parameter Learning Explained. *CoRR, abs/1411.2738.*
- [6] Ma, Y., Rusu, F., & Torres, M. (2018). Stochastic Gradient Descent on Highly-Parallel Architectures. *CoRR, abs/1802.08800.*