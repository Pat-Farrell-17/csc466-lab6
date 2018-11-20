#   Authors:    Ryan Nett
#               Patrick Farrell
#   Course:     CSC 466 - Fall
#   Instructor: Dr. Dekhtyar
#   Program   : Lab 6 (Collaborative Filtering)
#
#   This file contains all methods related to reading the jester dataset in
#   and any related computations.
#
#   Data is read in as a pandas DataFrame, returned by readJester().
#   If a user has not rated a joke, the rating is stored as NaN rather than 99
#   so that it doesn't affect any calculations
#
#   All methods that return statistics about the data expect the data
#   as the DataFrame of the whole data set (parameter df) and either a
#   userID (the row that the user's ratings are in) or a jokeID/itemID
#   (the column containing all ratings for that joke/item).

import pandas as pd


# Reads the data file into a pandas DataFrame
def readJester(fname):
    return pd.read_csv(fname, na_values=[99])


def rating(jokeID, userID, df):
    return df.iloc[userID, jokeID + 1]


def jokeRatings(jokeID, df):
    return df.iloc[:, jokeID + 1]


def userRatings(userID, df):
    return df.iloc[:, 1:101].iloc[userID]

# Given a joke ID and a pandas DataFrame, returns the average rating of the
# joke for users that rated that joke
def avgJokeRating(jokeID, df):
    return df.iloc[:,jokeID].mean()

# Given a user ID and a pandas DataFrame, returns the average rating that
# the user gave the jokes that they rated
def avgUserRating(userID, df):
    return df.iloc[:,1:101].iloc[userID].mean()

def main():
    # os.chdir(r"C:\Users\Ian\Documents\CSC466\Lab 6")
    data = readJester("data/jester-data-1H.csv")
    print(data.head())
    print()

if __name__ == "__main__":
    main()
