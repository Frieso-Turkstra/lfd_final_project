import argparse
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import json
import logging
from utils import read_corpus


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--validation_file', type=str, required=True,
                        help='Validation file to evaluate on')
    parser.add_argument('-i', '--input_file', type=str, required=True,
                        help='File containing the predictions')
    parser.add_argument('-l', '--log_file', type=str, default='../logs/evaluate.log',
                        help='Log file to save report and confusion matrix')
    
    args = parser.parse_args()
    return args
    

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


if __name__ == "__main__":
    args = create_arg_parser()

    # Read in the true labels and predictions
    messages, labels = read_corpus(args.validation_file)
    predictions = json.load(open(args.input_file))

    # Retrieve the classification report and confusion matrix
    report, error_matrix = evaluate(labels, predictions)
    print(report)

    # Log the scores and confusion matrix 
    logging.basicConfig(filename=args.log_file, encoding='utf-8', level=logging.DEBUG)
    logging.info(f'Classification report:\n{report}')
    logging.info(f'Confusion matrix:\n{error_matrix}')
