#   Authors:    Ryan Nett
#               Patrick Farrell
#   Course:     CSC 466 - Fall
#   Instructor: Dr. Dekhtyar
#   Program   : Lab 6 (Collaborative Filtering)

import evaluate as ev
from sys import argv

def readTests(testfile):
    tests = []
    with open(testfile, "r") as f:
        for line in f.readlines():
            tests.append([int(i) for i in line.split(',')])
    return tests

# If no arguments, print help message
# Otherwise, run randomSample with passed args
def main(args):
    if len(args) == 1:
        print("Use the following IDs for each method\n")
        print("0 : item-based, no adjust, no knn, cosine similarity\n"+
               "1 : user-based, no adjust, no knn, cosine similarity\n"+
               "2 : item-based, adjusted, no knn, cosine similarity\n"+
               "3 : user-based, adjusted, no knn, cosine similarity\n"+
               "4 : item-based, no adjust, knn, cosine similarity\n"+
               "5 : user-based, no adjust, knn, cosine similarity\n"+
               "6 : item-based, adjusted, knn, cosine similarity\n"+
               "7 : user-based, adjusted, knn, cosine similarity\n"+
               "8 : item-based, no adjust, no knn, pearson similarity\n"+
               "9 : user-based, no adjust, no knn, pearson similarity\n"+
               "10 : item-based, adjusted, no knn, pearson similarity\n"+
               "11 : user-based, adjusted, no knn, pearson similarity\n"+
               "12 : item-based, no adjust, knn, pearson similarity\n"+
               "13 : user-based, no adjust, knn, pearson similarity\n"+
               "14 : item-based, adjusted, knn, pearson similarity\n"+
               "15 : user-based, adjusted, knn, pearson similarity")
        print("\nUse k = 0 for no knn")
        print("\nCall: python EvaluateCFRandom.py <MethodID> <Size> <Repeats> <k>")
    else:
        ev.accuracyMeasures(ev.userTest(int(args[1]) ,readTests(args[2]),
            int(args[3])))

if __name__ == "__main__":
    main(argv)
