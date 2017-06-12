#!/usr/bin/python

from time import time
from datetime import datetime

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''

    # Start the clock
    start = time()

    # train the classifier
    clf.fit(X_train, y_train)

    # Stop the clock
    end = time()

    # Print timing
    print ("Trained model in {:.4f} seconds".format(end - start))

def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''

    # Start the clock
    start = time()

    # make predictions
    y_pred = clf.predict(features)

    # Stop the clock
    end = time()

    # Print and return results
    print ("Made predictions in {:.4f} seconds.".format(end - start))

    return f1_score(target, y_pred, pos_label=1)

def predict_labels_accuracy(clf, features, target):
    ''' Makes predictions using a fit classifier based on the accuracy score. '''

    y_pred = clf.predict(features)

    # Print and return results
    return accuracy_score(target, y_pred)

def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''

    # Indicate the classifier and the training set size
    print ("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))

    # Train the classifier
    train_classifier(clf, X_train, y_train)

    # Print the results of prediction for both training and testing
    print ("F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))
    print ("Accuracy score for training set: {:.4f}.".format(predict_labels_accuracy(clf, X_train, y_train)))
    print ("F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test)))
    print ("Accuracy score for test set: {:.4f}.".format(predict_labels_accuracy(clf, X_test, y_test)))
