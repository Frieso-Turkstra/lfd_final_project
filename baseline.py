'''
This script uses the Naive Bayes classifier with a simple bag-of-words vectorizer. 
'''

import argparse
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
import pandas as pd
import numpy as np


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-tf', '--train_file', default='data/train.tsv', type=str,
                        help='Train file to learn from (default train.tsv)')
    parser.add_argument('-df', '--dev_file', default='data/dev.tsv', type=str,
                        help='Dev file to evaluate on (default dev.tsv)')
    
    args = parser.parse_args()
    return args


def read_corpus(corpus_file):
    '''Reads a string and returns a list of words and a list of corresponding labels.'''
    documents = []
    labels = []

    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.strip().split('\t')
            documents.append(tokens[0])
            labels.append(tokens[1])

    return documents, labels


def evaluate(Y_test, Y_pred):
    ''' Takes in true labels Y_test, predicted labels Y_pred.
    Returns a classification report (precision, recall
    and F1 score for each class) and a confusion matrix with labels.'''
    report = classification_report(Y_test, Y_pred)

    # Create a confusion matrix with labels
    labels = np.unique(Y_test)
    cm = confusion_matrix(Y_test, Y_pred, labels=labels)
    cm_labeled = pd.DataFrame(cm, index=labels, columns=labels)

    return report, cm_labeled


def grid_search(classifier, param_grid, cv=3):
    """Takes as input a classifier model or pipeline, a parameter grid in the form
    of a dictionary with parameters as keys and a list of values as value,
    and a value cv for cross validation.
    Make sure the keys start with 'cls__' when they are used in a Pipeline"""
    return GridSearchCV(classifier, param_grid, n_jobs=-1, cv=cv, refit=True, scoring='f1_macro')


def identity(inp):
    '''Dummy function that just returns the input'''
    return inp


if __name__ == '__main__':
    args = create_arg_parser()

    # Create the test and training sets with appropriate labels
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)

    # Bag of Words vectorizer
    vec = CountVectorizer(preprocessor=identity, tokenizer=identity)

    classifier = Pipeline([('vec', vec), ('cls', MultinomialNB())])
    param_grid = {
        'cls__alpha': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
        'cls__fit_prior': [True, False],
    }
    grid = grid_search(classifier, param_grid)

    # Fit the specified classifier on the training data
    grid.fit(X_train, Y_train)

    # Use the fitted classifier to predict classes on the test data
    Y_pred = grid.predict(X_dev)

    # Retrieve the classification report and confusion matrix
    report, error_matrix = evaluate(Y_dev, Y_pred)
    
    # Log the best parameters and the corresponding scores and confusion matrix
    logging.basicConfig(filename=f'{classifier["cls"].__class__.__name__}.log', encoding='utf-8', level=logging.DEBUG)
    logging.info(f'Parameter grid:\n{param_grid}\n')
    logging.info(f'Best parameters:\n{grid.best_params_}\n')
    logging.info(f'Evaluation:\n{report}')
    logging.info(f'Confusion matrix:\n{error_matrix}')
