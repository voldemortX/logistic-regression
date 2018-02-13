import nltk
import pickle
import re
import numpy as np

def getX_ori():
    print("Fetching reviews...");
    # get a list of sentences in lowercase
    text = open("Reviews.txt").read();
X    pattern = re.compile(r'[^a-z\n ]+');
    text = pattern.sub('', text);
    X_ori = re.split("\n", text);
    return X_ori;


def getWordList():
    # get the word list
    print("Getting word list...");
    pkl_file = open('wordList.pkl', 'rb')
    wordList = pickle.load(pkl_file);
    pkl_file.close();
    return wordList;


def calculateFeatures(X_ori, wordList):
    # calculate features for X(m, n_x)
    print("Calculating features...");
    n_x = len(wordList);
    m = len(X_ori);
    X = np.zeros((m, n_x));
    for i in range(m):
        tempX = re.split(' ', X_ori[i]);
        for j in range(len(tempX)):
            tempX[j] = nltk.PorterStemmer().stem(tempX[j]);

        for j in range(n_x):
            if wordList[j] in tempX:
                X[i, j] = 1;

    np.save("X.npy", X);
    return X;


def divideData(X, y):
    # divide data by 60:20:20
    np.random.seed(1);  # stabilize outcome
    m = X.shape[0];
    X = np.insert(X, 0, values=np.squeeze(y), axis=1);
    np.random.shuffle(X);
    X = X.T;
    y = X[0, :].reshape(1, m);
    X = np.delete(X, 0, 0);
    X_train = X[:, 0:12000];
    X_cv = X[:, 12000:16000];
    X_test = X[:, 16000:];
    y_train = y[:, 0:12000].reshape(1, 12000);
    y_cv = y[:, 12000:16000].reshape(1, 4000);
    y_test = y[:, 16000:].reshape(1, 4000);
    return X_train, X_cv, X_test, y_train, y_cv, y_test;


def gety():
    print("Fetching labels...");
    # get labels in a vector
    text = open("Labels.txt").read();
    y_ori = re.split("\n", text);
    m = len(y_ori);
    y = np.zeros((m, 1));
    for i in range(20000):
        if y_ori[i] is "positive":
            y[i, :] = 1;

    np.save("y.npy", y);
    return y;

def pre():
    # process raw data
    y = gety();
    X_ori = getX_ori();
    wordList = getWordList();
    X = calculateFeatures(X_ori, wordList);
    X_train, X_cv, X_test, y_train, y_cv, y_test = divideData(X, y);
    return X_train, X_cv, X_test, y_train, y_cv, y_test;


def fastPre():
    # this function processes raw data real quick when you already have X and y stored
    y = np.load("y.npy");
    X = np.load("X.npy");
    X_train, X_cv, X_test, y_train, y_cv, y_test = divideData(X, y);
    return X_train, X_cv, X_test, y_train, y_cv, y_test;
