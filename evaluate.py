#   Authors:    Ryan Nett
#               Patrick Farrell
#   Course:     CSC 466 - Fall
#   Instructor: Dr. Dekhtyar
#   Program   : Lab 6 (Collaborative Filtering)

import processing as ps
from predict import predict, transformMatrix
from sys import argv

# Generates <size> random test cases which are tested <repeats> time on
# the specified method
# Implemented methods:
#           KNN, cosine/pearson similarity, adjusted/non adjusted weighted avg,
#           user/item based
#
# Method IDs:
#           0 : item-based, no adjust, no knn, cosine similarity
#           1 : user-based, no adjust, no knn, cosine similarity
#           2 : item-based, adjusted, no knn, cosine similarity
#           3 : user-based, adjusted, no knn, cosine similarity
#           4 : item-based, no adjust, knn, cosine similarity
#           5 : user-based, no adjust, knn, cosine similarity
#           6 : item-based, adjusted, knn, cosine similarity
#           7 : user-based, adjusted, knn, cosine similarity
#           8 : item-based, no adjust, no knn, pearson similarity
#           9 : user-based, no adjust, no knn, pearson similarity
#          10 : item-based, adjusted, no knn, pearson similarity
#          11 : user-based, adjusted, no knn, pearson similarity
#          12 : item-based, no adjust, knn, pearson similarity
#          13 : user-based, no adjust, knn, pearson similarity
#          14 : item-based, adjusted, knn, pearson similarity
#          15 : user-based, adjusted, knn, pearson similarity
def randomSample(method, size, repeats, k):
    jokefile = "data/jester-data-1.csv"
    outfilebase = "evalRandom"
    mtrx = ps.readJester(jokefile)

    user = bool(method % 2)
    adjust = True if method in [2,3,6,7,10,11,14,15] else False
    pearson = True if method > 7 else False
    m = transformMatrix(mtrx.values, user)

    print("userID\titemID\tActual_Rating\tPredicted_Rating\tDelta_Rating")
    predictions = []
    for i in range(repeats):
        predRun = []
        tests = ps.selectTests(mtrx, size)
        for uid, jokeid in tests:
            pred, act = predict(m, uid, jokeid, user, adjust, k, pearson)
            predRun.append((uid, jokeid, pred, act))
            print("{0:3d}\t{1:3d}\t{2:.2f}\t\t{3:.2f}\t\t\t{4:.2f}".format(uid,
                    jokeid, pred, act, pred-act))
        print("MAE: {0:.3f}\n".format(mae(predRun)))
        predictions += predRun

    return predictions


# Expects a list of (uid, jokeid) to test
# Size parameter is ignored
def userTest(method, tests, k):
    jokefile = "data/jester-data-1.csv"
    mtrx = ps.readJester(jokefile)
    user = bool(method % 2)
    adjust = True if method in [2,3,6,7,10,11,14,15] else False
    pearson = True if method > 7 else False
    m = transformMatrix(mtrx.values, user)

    print("userID\titemID\tActual_Rating\tPredicted_Rating\tDelta_Rating")

    predictions = []
    for uid, jokeid in tests:
        pred, act = predict(m, uid, jokeid, user, adjust, k, pearson)
        predictions.append((uid, jokeid, pred, act))
        print("{0:3d}\t{1:3d}\t{2:.2f}\t\t{3:.2f}\t\t\t{4:.2f}".format(uid,
                jokeid, pred, act, pred-act))
    print("MAE: {0:.3f}\n".format(mae(predictions)))
    return predictions

# Mean Absolute Error
def mae(predictions):
    return sum(abs(pred - act) for _, _, pred, act in predictions) / len(predictions)

def accuracyMeasures(predictions):
    tp, tn, fp, fn = 0, 0, 0, 0
    for _, _, pred, act in predictions:
        if pred >= 5.0 and act >= 5.0:
            tp += 1
        elif pred >= 5.0 and act < 5.0:
            fp += 1
        elif pred < 5.0 and act >= 5.0:
            fn += 1
        else:
            tn += 1

    print("\t\tRecommended\tNot Recommended")
    print("Relevant\t{}\t\t{}".format(tp, fn))
    print("Irrelevant\t{}\t\t{}".format(fp, tn))

    if tp == 0:
        precision = 0.0
        recall = 0.0
        f1m = 0.0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1m = 2 * precision * recall / (precision + recall)
    print("Precision : {}\nRecall : {}\nF1-Measure : {}".format(precision,
                                                                recall,
                                                                f1m))
    print("Total Accuracy : ", (tp + tn) / (tp + tn + fp + fn))
    return precision, recall, f1m

def main(args):
    method = int(args[1])
    size = int(args[2])
    repeats = int(args[3])
    if len(args == 5):
        k = int(args[4])
    else:
        k = 0
    accuracyMeasures(randomSample(method, size, repeats, k))

if __name__ == "__main__":
    main(argv)
