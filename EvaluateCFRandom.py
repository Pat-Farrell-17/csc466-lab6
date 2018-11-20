#   Authors:    Ryan Nett
#               Patrick Farrell
#   Course:     CSC 466 - Fall
#   Instructor: Dr. Dekhtyar
#   Program   : Lab 6 (Collaborative Filtering)

import processing as proc
import sys


# User-based mean utility method
#
# Input:    df      - ratings matrix
#           jokeid  - joke number to predict for
# Output:   predicted joke rating
#           (same predicted rating for any user, so uid doesn't matter)
def meanU(df, jokeid):
    return proc.avgJokeRating(jokeid, df)

# Item-based mean utility method
#
# Input:    df      - ratings matrix
#           uid     - user to predict for
# Output:   predicted joke rating for the given user
#           (same predicted rating for any joke rated by the user, so
#           jokeid doesn't matter)
def meanI(df, uid):
    return proc.avgUserRating(uid, df)

# Item-based weighted sum method
#
# Input:    df      - ratings matrix
#           uid     - user to predict for
#           jokeid  - joke number to predict for
# Output:   predicted joke rating for user the given user
def weightedSumI(df, uid, jokeid, sim):
    k, wSum = 0, 0
    for uidOther, _ in df.iterrows():
        k += abs(sim(df, uid, uidOther))
        wSum += sim(df, uid, uidOther) * proc.getJokeRating(df, uid, jokeid)

    return (1 / k) * wSum

# User-based weighted sum method
#
# Input:    df      - ratings matrix
#           uid     - user to predict for
#           jokeid  - joke number to predict for
# Output:   predicted joke rating for user the given user
def weightedSumU(df, uid, jokeid, sim):
    k, wSum = 0, 0
    for uidOther, _ in df.iterrows():
        k += abs(sim(df, uid, uidOther))
        wSum += sim(df, uid, uidOther) * proc.getJokeRating(df, uid, jokeid)

    return (1 / k) * wSum



def main(args):
    data = proc.readJester("data/jester-data-1H.csv")
    print(weightedSumI(data, 0, 1, proc.cossimI))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please call with args")
    else:
        main(sys.argv)
