import timeit
import numpy as np


def create_setup(size):
    return """
from random import randrange, choices
import numpy as np

SIZE = {}

weights = np.random.rand(SIZE)
s = np.sum(weights)
weights /= s
arr = list(range(SIZE))
""".format(size)


random_style = """
cum_weights = np.cumsum(weights)
choices(arr, cum_weights=cum_weights, k=50)
"""

numpy_style = """
np.random.choice(SIZE, 50, p=weights)
"""


def timer(s, v='', nloop=10000, nrep=3):
    units = ["s", "ms", "µs", "ns"]
    scaling = [1, 1e3, 1e6, 1e9]
    Timer = timeit.Timer(stmt=s, setup=v)
    best = min(Timer.repeat(nrep, nloop)) / nloop
    if best > 0.0:
        order = min(-int(np.floor(np.log10(best)) // 3), 3)
    else:
        order = 3
    print("%d loops, best of %d: %.*g %s per loop" % (nloop, nrep,
                                                      3,
                                                      best * scaling[order],
                                                      units[order]))


for k in [100, 1000, 10000, 100000, 500000]:
    print("Random", k)
    timer(random_style, create_setup(k))
    print("Numpy", k)
    timer(numpy_style, create_setup(k))


# random.choices is significantly faster when both functions are used in optimal conditions
# random.choices is still faster than numpy.choices if we perform the cumsum with numpy right before the function call
