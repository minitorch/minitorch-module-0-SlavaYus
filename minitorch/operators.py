"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

EPS1 = 1e-2
EPS2 = 1e-7

def mul(x: float, y: float) -> float:
    """Multiply"""
    return x * y

def id(x: float) -> float:
    """Id"""
    return x

def add(x: float, y: float) -> float:
    """Addition"""
    return x + y

def neg(x: float) -> float:
    """Negative"""
    return -x

def lt(x: float, y: float) -> float:
    """Less Than"""
    if x < y:
        return 1.0
    else:
        return 0.0

def eq(x: float, y: float) -> float:
    """Equal"""
    if x == y:
        return 1.0
    else:
        return 0.0

def max(x: float, y: float) -> float:
    """Maximum"""
    if x > y:
        return x
    else:
        return y

def is_close(x: float, y: float) -> float:
    """Is Close"""
    return abs(x - y) < EPS1

def sigmoid(x: float) -> float:
    """Sigmoid"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        math.exp(x) / (1.0 + math.exp(x))

def relu(x: float) -> float:
    """ReLU"""
    if x >= 0:
        return x
    else:
        return 0

def log(x: float) -> float:
    """Log"""
    return math.log(x + EPS2)

def exp(x: float) -> float:
    """Exponent"""
    return math.exp(x)

def log_back(x: float, y: float) -> float:
    """Log Back"""
    return y / (x + EPS2)

def inv(x: float) -> float:
    """Inverse"""
    return 1.0 / (x)

def inv_back(x: float, y: float) -> float:
    """Inverse Back"""
    return -y / (x**2)

def relu_back(x: float, y: float) -> float:
    """ReLU Back"""
    if x > 0:
        return y
    else:
        return 0


# Task 3

def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """MAP High Order"""

    def map1(mas: Iterable[float]) -> Iterable[float]:
        res = []
        for x in mas:
            res += [fn(x)]
        return res

    return map1

def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """ZIP High Order"""

    def zipWith1(mas1: Iterable[float], mas2: Iterable[float]) -> Iterable[float]:
        res = []
        for x, y in zip(mas1, mas2):
            res += [fn(x, y)]
        return res
    return zipWith1


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order reduce
    Args:
    fn: Function from two values to one value
    Returns:
    A function that takes a list, an initial value, and reduces the list to a single value using fn.
    """

    def _reduce(mas: Iterable[float]) -> float:
        val = start
        for u in mas:
            val = fn(val, u)
        return val

    return _reduce

def negList(mas: Iterable[float]) -> Iterable[float]:
    return map(neg)(mas)

def addLists(mas1: Iterable[float], mas2: Iterable[float]) -> Iterable[float]:
    return zipWith(add)(mas1, mas2)

def sum(mas: Iterable[float]) -> float:
    return reduce(add, 0)(mas)

def prod(mas: Iterable[float]) -> float:
    return reduce(mul, 1)(mas)

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
