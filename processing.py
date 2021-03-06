#   Authors:    Ryan Nett
#               Patrick Farrell
#   Course:     CSC 466 - Fall
#   Instructor: Dr. Dekhtyar
#   Program   : Lab 6 (Collaborative Filtering)
#
#   This file contains all methods related to reading the jester dataset in
#   and any related computations or sampling.
#
#   Data is read in as a pandas DataFrame, returned by readJester().
#   If a user has not rated a joke, the rating is stored as NaN rather than 99
#   so that it doesn't affect any calculations
#
#   All methods that return statistics about the data expect the data
#   as the DataFrame of the whole data set (parameter df) and either a
#   uid (the row that the user's ratings are in) or a jokeid/jokeID
#   (the column containing all ratings for that joke/item).

import math
import os
from random import randint

import numpy as np
import pandas as pd


# Reads the data file into a pandas DataFrame
def readJester(fname):
    df = pd.read_csv(fname,header=None, na_values=[99])
    df.drop(df.columns[0], axis=1, inplace=True)
    df.set_axis([i for i in range(df.shape[1])], axis='columns', inplace=True)
    return df

def buildCossimMatrix(df):
    return df.replace([np.nan], [0.0])

# Given a joke ID and a pandas DataFrame, returns the average rating of the
# joke for users that rated that joke
def avgJokeRating(jokeID, df):
    return df.iloc[:,jokeID].mean()

# Given a user ID and a pandas DataFrame, returns the average rating that
# the user gave the jokes that they rated
def avgUserRating(uid, df):
    return df.iloc[uid].mean()

# Given the ratings matrix df, a uid and a jokeid returns the rating that the
# user gave that joke, or 0 if the user didn't rate it
def rating(jokeid, uid, df):
    if df.isnull().ix[uid, jokeid]:
        return 0
    else:
        return df.iloc[uid, jokeid]

# Returns a random jokeid given a DataFrame and a uid
# df.iloc[uid, jokeid] is guaranteed to be valid
# i.e. the user at uid has rated the joke at jokeid
def getRandJoke(df, uid):
    jokeid = randint(0, 99)
    while df.isnull().ix[uid, jokeid]:
        jokeid = randint(0, 99)
    return jokeid

# Returns a list of size random (userID, jokeID) tuples from the given DataFrame
# where the user is guaranteed to have rated the joke
def selectTests(df, size):
    s = df.sample(size)
    tests = []
    for i in s.index:
        colID = getRandJoke(df, i)
        tests.append((i, colID))
    return tests

# Given the ratings matrix df and two uids, calculate the cosine similarity
# between their corresponding ratings vectors
def cossimI(df, uidA, uidB):
    vecA = df.iloc[uidA]
    vecB = df.iloc[uidB]
    return sum(r1 * r2 for r1, r2 in zip(vecA, vecB) if (pd.notnull(r1) and
                pd.notnull(r2))) / (
                math.sqrt(sum(r **2 for r in vecA if pd.notnull(r))) *
                math.sqrt(sum(r **2 for r in vecB if pd.notnull(r))))

def cossimU(df, jidA, jidB):
    vecA = df.iloc[:,jidA]
    vecB = df.iloc[:,jidB]
    return sum(r1 * r2 for r1, r2 in zip(vecA, vecB) if (pd.notnull(r1) and
                pd.notnull(r2))) / (
                math.sqrt(sum(r **2 for r in vecA if pd.notnull(r))) *
                math.sqrt(sum(r **2 for r in vecB if pd.notnull(r))))
def main():
    os.chdir(r"C:\Users\Ian\Documents\CSC466\Lab 6\trunk")
    data = readJester("data/jester-data-1.csv")
    print(data.head())
    print(data.shape)

if __name__ == "__main__":
    main()
