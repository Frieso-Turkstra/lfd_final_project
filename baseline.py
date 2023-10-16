'''
High-level description of the program
'''

import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


def create_arg_parser():
    parser = argparse.ArgumentParser()

    # Command line arguments to choose the training and development files
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


def print_measures(Y_test, Y_pred, plot_cm=False):
    ''' Takes in true labels Y_test, predicted labels Y_pred. 
    Prints a classification report (precision, recall
    and F1 score for each class)'''
    report = classification_report(Y_test, Y_pred)
    print("Classification report:\n\n", report)


def identity(inp):
    '''Dummy function that just returns the input'''
    return inp


if __name__ == '__main__':
    args = create_arg_parser()

    # Create the test and training sets with appropriate labels
    X_train, Y_train = read_corpus(args.train_file)
    X_test, Y_test = read_corpus(args.dev_file)

    # Bag of Words vectorizer
    vec = CountVectorizer(preprocessor=identity, tokenizer=identity, ngram_range=(1,3), min_df=3)

    classifier = Pipeline([('vec', vec), ('cls', MultinomialNB())])

    # Fit the specified classifier on the training data
    classifier.fit(X_train, Y_train)

    # Use the fitted classifier to predict classes on the test data
    Y_pred = classifier.predict(X_test)

    # Print classification report
    print_measures(Y_test, Y_pred)
