from math import isnan
from sys import argv

from numpy import nanmean

from distance import cossim, pearson
from frequencyMatrix import FrequencyMatrix


def predict(fm: FrequencyMatrix, userID: int, jokeID: int, byUser: bool, adjust: bool, k: int, usePearson: bool):
    original = fm.users[userID][jokeID]
    fm.dataFrame.set_value(userID, jokeID + 1, float('nan'))

    if byUser:
        itemID = userID
        otherID = jokeID
        items = fm.users
    else:
        itemID = jokeID
        otherID = userID
        items = fm.jokes

    list = items[itemID]

    similarities = {}
    means = {}

    means[itemID] = nanmean(items[itemID])

    for i in range(0, len(items)):
        if i != itemID:
            if usePearson:
                other = items[i]
                means[i] = nanmean(other)
                similarities[i] = pearson(list, other, means[itemID], means[i])
            else:
                other = items[i]
                similarities[i] = cossim(list, other)

    if k <= 0:
        use = set(similarities.keys())
    else:
        nearest = sorted([(sim, idx) for idx, sim in similarities.items()], reverse=True)
        use = set([elem[1] for elem in nearest[:k]])

    useSims = {k: v for k, v in similarities.items() if k in use}

    factor = 1 / sum(abs(s) for s in useSims.values())

    if adjust:
        topSum = 0
        for idx, sim in useSims.items():
            if idx not in means:
                means[idx] = nanmean(items[idx])

            rate = items.rating(idx, otherID)
            if not isnan(rate):
                topSum += sim * (rate - means[idx])

        score = means[itemID] + factor * topSum
    else:
        topSum = 0
        for idx, sim in useSims.items():
            rate = items.rating(idx, otherID)
            if not isnan(rate):
                topSum += sim * rate

        score = factor * topSum

    return score, original


# user=false -> items, k=0 -> no knn, pearson=false -> cosine
def main(jesterFile: str, userID: int, jokeID: int, user: bool, adjust: bool, k: int, pearson: bool):
    predicted, original = predict(FrequencyMatrix(jesterFile), userID, jokeID, user, adjust, k, pearson)
    print("Original: ", original)
    print("Predicted: ", predicted)


if __name__ == '__main__':
    main(argv[1], int(argv[2]), int(argv[3]), bool(int(argv[4])), bool(int(argv[5])), int(argv[6]), bool(int(argv[7])))
