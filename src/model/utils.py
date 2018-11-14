import functools
import operator

def prod(l):
    return functools.reduce(operator.mul, l, 1)
