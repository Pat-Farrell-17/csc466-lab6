import processing as ps


class UserRatings:
    def __init__(self, fm):
        self.frequencyMatrix = fm
        self.length = len([s[1] for s in self.frequencyMatrix.dataFrame.iloc[:, 1:101].iterrows()])

    def __getitem__(self, item):
        return [s[1] for s in self.frequencyMatrix.dataFrame.iloc[item, 1:101].iteritems()]

    def __iter__(self):
        return iter([s[1] for s in self.frequencyMatrix.dataFrame.iloc[:, 1:101].iterrows()])

    def __len__(self):
        return self.length

    def rating(self, userID, jokeID):
        return self.frequencyMatrix.getRating(jokeID, userID)


class JokeRatings:
    def __init__(self, fm):
        self.frequencyMatrix = fm
        self.length = len([s[1] for s in self.frequencyMatrix.dataFrame.iloc[:, 1:101].iteritems()])

    def __getitem__(self, item):
        return [s[1] for s in self.frequencyMatrix.dataFrame.iloc[:, item + 1].iteritems()]

    def __iter__(self):
        return iter([s[1] for s in self.frequencyMatrix.dataFrame.iloc[:, 1:101].iteritems()])

    def __len__(self):
        return self.length

    def rating(self, jokeID, userID):
        return self.frequencyMatrix.getRating(jokeID, userID)

class FrequencyMatrix:
    def __init__(self, jokeFile):
        self.dataFrame = ps.readJester(jokeFile)

        self.jokes = JokeRatings(self)
        self.users = UserRatings(self)

    def getRating(self, jokeID, userID):
        return ps.rating(jokeID, userID, self.dataFrame)


if __name__ == '__main__':
    fm = FrequencyMatrix("data/jester-data-1H.csv")
