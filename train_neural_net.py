#!/usr/bin/env python

from __future__ import division
import os

import pandas as pd
import numpy as np

from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.layers import Dropout

import prepare_data

FIRST_HIDDEN_OUTPUT_DIMENSION = 280
SECOND_HIDDEN_OUTPUT_DIM = 300
NUM_EPOCHS = 200
NUM_TRAINING_SAMPLES = 6273
STD_DEV = 1 / np.sqrt(NUM_TRAINING_SAMPLES)

#Number of training folds to use (1-10) - the rest will be used for testing
NUM_TRAINING_FOLDS = 7


def one_hot_encode(labels):
    """
    Returns one hot encoding of the labels passed in. Taken from
    http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/
    """
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels.T] = 1
    return one_hot_encode


def load_data(filename):
    X = pd.read_csv(header=None, usecols=range(0,193), filepath_or_buffer=filename, index_col=False)
    y = pd.read_csv(header=None, usecols=[193], filepath_or_buffer=filename, index_col=False)
    return X.values, y.values

def train_neural_net():
    """
    Trains a four-layer neural network with our training data.
    """
    model = Sequential([
        Dense(output_dim=FIRST_HIDDEN_OUTPUT_DIMENSION, input_dim=193),
        Activation("tanh"),
        Dropout(0.5),
        Dense(output_dim=SECOND_HIDDEN_OUTPUT_DIM),
        Activation("sigmoid"),
        Dropout(0.5),
        Dense(output_dim=10),
        Activation("softmax")
    ])
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    X, y = load_data("./train_data.csv")

    y = one_hot_encode(y.astype(int))
    model.fit(X, y, nb_epoch=NUM_EPOCHS, verbose=2)
    return model


def test_neural_net(trained_model):
    correct_labels = 0
    incorrect_labels = 0
    X_test, y_test = load_data("./test_data.csv")
    predictions = trained_model.predict_classes(X_test, batch_size=32)
    for idx, label in enumerate(y_test.T[0].astype(int)):
        if predictions[idx] == label:
            correct_labels += 1
        else:
            incorrect_labels += 1
    accuracy = correct_labels / (correct_labels + incorrect_labels)
    return accuracy

if __name__ == "__main__":
    #If the prepared data files don't exist, create them.
    if not (os.path.isfile("./train_data.csv") or os.path.isfile("./test_data.csv")):
        prepare_data.process_csv(8)
    model = train_neural_net()
    accuracy = test_neural_net(model)
    print "Test set accuracy: {}".format(accuracy)
    


