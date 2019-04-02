import ctypes
from multiprocessing import Pool, Array, Value
import numpy as np


def emb2Arr(emb):
    """Converts embedding to a shared array."""
    return Array(ctypes.c_double, emb.ravel())


def arr2Arr(arr, is_int=False):
    """Converts np.ndarray to a shared array."""
    if is_int:
        return Array(ctypes.c_int, arr, lock=False)
    return Array(ctypes.c_double, arr, lock=False)


def int2Val(value):
    """Converts an int to a shared int."""
    return Value('i', value, lock=False)


def Arr2emb(arr, v=None):
    """Reads a shared array as embeddings, without data copy by leveraging on some numpy magic."""
    global VOCABSIZE
    if v is None:
        v = VOCABSIZE.value
    return np.frombuffer(arr.get_obj()).reshape((v, -1))


def Arr2arr(arr, is_int=False):
    """Reads a shared array as a np.array, without data copy by leveraging on some numpy magic."""
    if is_int:
        return np.frombuffer(arr, dtype="int32")
    return np.frombuffer(arr)


def init(vocab_, k_, context_size_, noise_probas_, w_emb_, c_emb_, all_ids_):
    """Little helper to share the data between all workers."""
    global VOCABSIZE, k_factor, context_size, noise_probas, word_embeddings, context_embeddings, probas, all_ids
    VOCABSIZE = vocab_
    k_factor = k_
    context_size = context_size_
    noise_probas = noise_probas_
    word_embeddings = w_emb_
    context_embeddings = c_emb_
    all_ids = all_ids_


def parallel_iter(fun, args, n_worker, initargs):
    p = Pool(n_worker, initializer=init, initargs=initargs)
    res = p.imap_unordered(fun, args, chunksize=500)  # small chunks to get some speed improvement
    for r in res:
        yield r
    p.close()
    p.join()


def unpack_iterator(iterator):
    while True:
        listed_res = next(iterator)
        for res in listed_res:
            if res:
                yield res


def add_to_iterator(iterator, arg):
    for d in iterator:
        yield((d, arg))


def build_iterator(fun, args, eta, n_worker, initargs):
    base_iterator = parallel_iter(fun, args, n_worker, initargs)
    unpacked_iterator = unpack_iterator(base_iterator)
    final_iterator = add_to_iterator(unpacked_iterator, eta)
    return final_iterator
