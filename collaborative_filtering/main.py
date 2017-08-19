# libraries
import numpy as np
from scipy import io

# files
import utils
from user import User

ITEM_LIST = []

Y = None
R = None

if __name__ == '__main__':
    # fill ITEM_LIST, Y and R
    # get list of movies from text file
    f = open('movie_ids.txt', 'r', encoding='ISO-8859-1')
    for l in f.readlines():
        movie = l.split(' ', 1)[1].split('\n')[0]
        ITEM_LIST.append(movie)

    # load the movie ratings dataset
    matDict = io.loadmat('ex8_movies.mat')
    # Y: no. items x no. users
    # R: same shape as Y, R(i,j) = 1 if and only if user j gave a rating to movie i
    Y = matDict['Y']
    R = matDict['R']
    print("Y: {} items x {} users".format(Y.shape[0], Y.shape[1]))

    # make recommendations for a new user
    user = User(ITEM_LIST)

    user.ratings[0] = 4
    user.ratings[97] = 2
    user.ratings[6] = 3
    user.ratings[11]= 5
    user.ratings[53] = 4
    user.ratings[63]= 5
    user.ratings[65]= 3
    user.ratings[68] = 5
    user.ratings[182] = 4
    user.ratings[225] = 5
    user.ratings[354]= 5

    user.printRatings(ITEM_LIST)
    user.train(Y, R)
    user.predict(ITEM_LIST, Y, R)
    user.printTop()
