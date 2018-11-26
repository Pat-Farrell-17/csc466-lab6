#   Authors:    Ryan Nett
#               Patrick Farrell
#   Course:     CSC 466 - Fall
#   Instructor: Dr. Dekhtyar
#   Program   : Lab 6 (Collaborative Filtering)

import processing as ps
from predict import predict
from frequencyMatrix import FrequencyMatrix
from sys import argv

# Generates <size> random test cases which are tested <repeats> time on
# the specified method
# Implemented methods:
#           KNN, cosine/pearson similarity, adjusted/non adjusted weighted avg,
#           user/item based
#
# Method IDs:
#           0 : user-based, no adjust, no knn, cosine similarity
#           1 : item-based, no adjust, no knn, cosine similarity
#           2 : user-based, adjusted, no knn, cosine similarity
#           3 : item-based, adjusted, no knn, cosine similarity
#           4 : user-based, no adjust, knn, cosine similarity
#           5 : item-based, no adjust, knn, cosine similarity
#           6 : user-based, adjusted, knn, cosine similarity
#           7 : item-based, adjusted, knn, cosine similarity

#           8 : user-based, no adjust, no knn, pearson similarity
#           9 : item-based, no adjust, no knn, pearson similarity
#          10 : user-based, adjusted, no knn, pearson similarity
#          11 : item-based, adjusted, no knn, pearson similarity
#          12 : user-based, no adjust, knn, pearson similarity
#          13 : item-based, no adjust, knn, pearson similarity
#          14 : user-based, adjusted, knn, pearson similarity
#          15 : item-based, adjusted, knn, pearson similarity
def randomSample(method: int, size: int, repeats: int):
    k = 50      # Parameterize?
    jokefile = "data/jester-data-1H.csv"
    mtrx = ps.readJester(jokefile)
    tests = ps.selectTests(mtrx, size)
    print("userID,itemID,Actual_Rating,Predicted_Rating,Delta_Rating")
    for i in range(repeats):
        predictions = []
        for uid, jokeid in tests:
            if method == 0:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            True, False, 0, False)
            elif method == 1:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            False, False, 0, False)
            elif method == 2:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            True, True, 0, False)
            elif method == 3:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            False, True, 0, False)
            elif method == 4:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            True, False, k, False)
            elif method == 5:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            False, False, k, False)
            elif method == 6:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            True, True, k, False)
            elif method == 7:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            False, True, k, False)
            elif method == 8:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            True, False, 0, True)
            elif method == 9:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            False, False, 0, True)
            elif method == 10:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            True, True, 0, True)
            elif method == 11:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            False, True, 0, True)
            elif method == 12:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            True, False, k, True)
            elif method == 13:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            False, False, k, True)
            elif method == 14:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            True, True, k, True)
            elif method == 15:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            False, True, k, True)
            else:
                print("Invalid method ID")
                return None
            predictions.append((uid, jokeid, pred, act))
            print(uid, jokeid, pred, act, pred - act )
        print(mae(predictions))

    return predictions


# Expects a list of (uid, jokeid) to test
# Size parameter is ignored
def userTest(method: int, size: int, repeats: int, testCandidates):
    k = 50      # Parameterize?
    jokefile = "data/jester-data-1H.csv"
    print("userID,itemID,Actual_Rating,Predicted_Rating,Delta_Rating")
    for i in range(repeats):
        predictions = []
        for uid, jokeid in tests:
            if method == 0:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            True, False, 0, False)
            elif method == 1:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            False, False, 0, False)
            elif method == 2:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            True, True, 0, False)
            elif method == 3:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            False, True, 0, False)
            elif method == 4:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            True, False, k, False)
            elif method == 5:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            False, False, k, False)
            elif method == 6:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            True, True, k, False)
            elif method == 7:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            False, True, k, False)
            elif method == 8:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            True, False, 0, True)
            elif method == 9:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            False, False, 0, True)
            elif method == 10:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            True, True, 0, True)
            elif method == 11:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            False, True, 0, True)
            elif method == 12:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            True, False, k, True)
            elif method == 13:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            False, False, k, True)
            elif method == 14:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            True, True, k, True)
            elif method == 15:
                pred, act = predict(FrequencyMatrix(jokefile),
                                            uid, jokeid,
                                            False, True, k, True)
            else:
                print("Invalid method ID")
                return None
            predictions.append((uid, jokeid, pred, act))
            print(uid, jokeid, pred, act, pred - act )
        print(mae(predictions))

    return predictions

# Mean Absolute Error
def mae(predictions):
    return sum(abs(pred - act) for pred, act in predictions) / len(predictions)

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
    print("Relevant\t{}\t{}".format(tp, fn))
    print("Irrelevant\t{}\t{}".format(fp, tn))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1m = 2 * precision * recall / (precision + recall)
    print("Precision : {}\nRecall : {}\nF1-Measure : {}".format(precision,
                                                                recall,
                                                                f1m))
    print("Total Accuracy : ", (tp + tn) / (tp + tn + fp + fn))
    return precision, recall, f1m

def main(args):
    randomSample(0, 50, 1)

if __name__ == "__main__":
    main(argv)
