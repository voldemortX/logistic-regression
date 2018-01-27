from processData import pre, fastPre
from logisticRegressionModels import *
import numpy as np
import re
import nltk
import pickle
import matplotlib as plt


# a review model by voldemort

def calculateWordList():
    # calculate word list
    text = open("Reviews.txt").read();
    pattern = re.compile(r'[^a-z\n ]+');
    text = pattern.sub('', text);
    wordList = [];
    counts = {};
    pattern = re.compile(r'[^a-z ]+');
    text = pattern.sub('', text);
    text = text.split(' ');
    print("Stemming...")
    for i in range(len(text)):
        text[i] = nltk.PorterStemmer().stem(text[i]);
    print("done");

    print("Counting frequency...");
    for word in text:
        if word in counts:
            counts[word] += 1;
        else:
            counts[word] = 1;
    print("Total words: " + str(len(counts)));

    print("Building wordList");
    for word in counts:
        if len(word) < 12:
            if word not in wordList and counts[word] > 50:
                wordList.append(word);
                print(word);

    print("Word list length: " + str(len(wordList)));

    # save to disk
    output = open('wordList.pkl', 'wb');
    pickle.dump(wordList, output, -1);
    output.close();


def reviewModel(num_iterations, learning_rate, print_cost):
    X_train, X_cv, X_test, y_train, y_cv, y_test = fastPre();
    print("Data is loaded!");
    n_x = X_train.shape[0];
    w, b = initializeWithZeros(n_x);
    parameters, grads, costs = optimize(w, b, X_train, y_train, num_iterations, learning_rate, print_cost);
    w = parameters["w"];
    b = parameters["b"];

    # predict
    y_prediction_cv = predict(w, b, X_cv);
    y_prediction_train = predict(w, b, X_train);
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100));
    print("cv accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_cv - y_cv)) * 100));