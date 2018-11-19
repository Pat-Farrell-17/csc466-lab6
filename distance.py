from math import isnan, sqrt


def cossim(a, b):
    topSum = 0
    bottomSum1 = 0
    bottomSum2 = 0

    if len(a) != len(b):
        raise ValueError('Arrays must be the same length')

    for i in range(0, len(a)):
        if not isnan(a[i]) and not isnan(b[i]):
            ai = a[i]
            bi = b[i]
            topSum += ai * bi
            bottomSum1 += ai ** 2
            bottomSum2 += bi ** 2

    return topSum / sqrt(bottomSum1 * bottomSum2)


def pearson(a, b, meanA, meanB):
    topSum = 0
    bottomSum1 = 0
    bottomSum2 = 0

    if len(a) != len(b):
        raise ValueError('Arrays must be the same length')

    for i in range(0, len(a)):
        if not isnan(a[i]) and not isnan(b[i]):
            ai = (a[i] - meanA)
            bi = (b[i] - meanB)
            topSum += ai * bi
            bottomSum1 += ai ** 2
            bottomSum2 += bi ** 2

    return topSum / sqrt(bottomSum1 * bottomSum2)
