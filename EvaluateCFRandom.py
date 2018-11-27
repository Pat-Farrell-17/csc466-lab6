#   Authors:    Ryan Nett
#               Patrick Farrell
#   Course:     CSC 466 - Fall
#   Instructor: Dr. Dekhtyar
#   Program   : Lab 6 (Collaborative Filtering)

from sys import argv

import evaluate as ev


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
        if int(args[1]) < 4 or (8 <= int(args[1]) < 12):
            ev.accuracyMeasures(ev.randomSample(int(args[1]), int(args[2]), int(args[3]), 0))
        else:
            ev.accuracyMeasures(ev.randomSample(int(args[1]), int(args[2]), int(args[3]), int(args[4])))

if __name__ == "__main__":
    main(argv)
