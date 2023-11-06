'''
This script implements five classic machine learning algorithms
to classify messages as offensive or not offensive:
- Naive Bayes
- Decision Tree 
- Random Forest
- K-nearest Neighbors
- Support Vector Machines (linear and non-linear)
The models are fitted on the training data and the predictions
on the test data are saved to an output file.
'''

import argparse
import logging
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from utils import read_corpus


def create_arg_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='classifier')
    
    # A subparser for the Naive Bayes Classifier
    parser_nb = subparsers.add_parser('nb', aliases=['naive_bayes'],
                                      help='Use Naive Bayes Classifier')
    parser_nb.add_argument('-a', '--alpha', type=float, default=10,
                           help='Set additive smoothing parameter')
    parser_nb.add_argument('-fp', '--fit_prior', type=bool, default=True,
                           help='Learn class prior probabilities')

    #  A subparser for the DecisionTreeClassifier
    parser_dt = subparsers.add_parser('dt', aliases=['decision_tree'],
                                      help='Use DecisionTreeClassifier')
    parser_dt.add_argument('-c', '--criterion', choices=['gini', 'entropy', 'log_loss'],
                           default='gini', help='Choose a criterion')
    parser_dt.add_argument('-s', '--splitter', choices=['random', 'best'],
                           default='best', help='Choose a splitter')
    parser_dt.add_argument('-md', '--max_depth', type=int, default=5,
                           help='Set maximum depth of a tree')
    parser_dt.add_argument('-mf', '--max_features', choices=['None', 'log2', 'sqrt'],
                           default='None', help='Set the number of features that are considered during a split')
    parser_dt.add_argument('-cw', '--class_weight', choices=['None', 'balanced'],
                           default='balanced', help='Assign class weights')

    # A subparser for the RandomForestClassifier
    parser_rf = subparsers.add_parser('rf', aliases=['random_forest'],
                                      help='Use RandomForestClassifier')
    parser_rf.add_argument('-c', '--criterion', choices=['gini', 'entropy', 'log_loss'],
                           default='gini', help='Choose a criterion')
    parser_rf.add_argument('-ne', '--n_estimators', type=int,
                           default=150, help='Choose the number of trees')
    parser_rf.add_argument('-md', '--max_depth', type=int,
                           default=5, help='Set maximum depth of a tree')
    parser_rf.add_argument('-mf', '--max_features', choices=['None', 'log2', 'sqrt'],
                           default='None', help='Set the number of features that are considered during a split')
    parser_rf.add_argument('-cw', '--class_weight', choices=['None', 'balanced', 'balanced_subsample'],
                           default='balanced', help='Assign class weights')
    
    # A subparser for the KNeighborsClassifier
    parser_kn = subparsers.add_parser('kn', aliases=['k_neighbors'],
                                      help='Use KNeighborsClassifier')
    parser_kn.add_argument('-nn', '--n_neighbors', type=int,
                           default=3, help='Set number of neighbors')
    parser_kn.add_argument('-w', '--weights', choices=['distance', 'uniform'],
                           default='distance', help='Choose a weight function')
    parser_kn.add_argument('-a', '--algorithm', choices=['brute', 'ball_tree', 'kd_tree'],
                           default='brute', help='Choose algorithm to compute the nearest neighbors')
    
    # A subparser for the SVC
    parser_svc = subparsers.add_parser('svc', aliases=['support_vector'],
                                       help='Use SVC')
    parser_svc.add_argument('-k', '--kernel', choices=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                            default='linear', help='Choose a kernel type')
    parser_svc.add_argument('-g', '--gamma', choices=['scale', 'auto'],
                            default='scale', help='Set kernel coefficient (only used for rbf, poly, sigmoid)')
    parser_svc.add_argument('-c', '--c', type=float,
                            default=1.0, help='Set regularization parameter')
    parser_svc.add_argument('-cw', '--class_weight', choices=['None', 'balanced'],
                           default='balanced', help='Assign class weights')
    
    # A subparser for the LinearSVC-Twitter
    parser_svcl_twitter = subparsers.add_parser('svcl_twitter', aliases=['support_vector_linear_twitter'],
                                        help='Use LinearSVC developed on Twitter dataset')
    parser_svcl_twitter.add_argument('-p', '--penalty', choices=['l1', 'l2'],
                             default='l1', help='Choose the norm used in penalization')
    parser_svcl_twitter.add_argument('-l', '--loss', choices=['hinge', 'squared_hinge'],
                             default='squared_hinge', help='Choose a loss function')
    parser_svcl_twitter.add_argument('-d', '--dual', type=bool,
                             default=False, help='Choose dual or primal optimization')
    parser_svcl_twitter.add_argument('-c', '--c', type=float,
                             default=0.1, help='Set regularization parameter')
    parser_svcl_twitter.add_argument('-cw', '--class_weight', choices=['None', 'balanced'],
                           default='balanced', help='Assign class weights')
    
    # A subparser for the LinearSVC-Telegram
    parser_svcl_telegram = subparsers.add_parser('svcl_telegram', aliases=['support_vector_linear_telegram'],
                                        help='Use LinearSVC developed on Telegram dataset')
    parser_svcl_telegram.add_argument('-p', '--penalty', choices=['l1', 'l2'],
                             default='l2', help='Choose the norm used in penalization')
    parser_svcl_telegram.add_argument('-l', '--loss', choices=['hinge', 'squared_hinge'],
                             default='hinge', help='Choose a loss function')
    parser_svcl_telegram.add_argument('-d', '--dual', type=bool,
                             default=True, help='Choose dual or primal optimization')
    parser_svcl_telegram.add_argument('-c', '--c', type=float,
                             default=1.0, help='Set regularization parameter')
    parser_svcl_telegram.add_argument('-cw', '--class_weight', choices=['None', 'balanced'],
                           default='balanced', help='Assign class weights')
    
    # Command line arguments to choose input and output files
    parser.add_argument('-t', '--train_file', type=str, required=True,
                        help='Train file to learn from')
    parser.add_argument('-v', '--validation_file', type=str, required=True,
                        help='Validation file to evaluate on')
    parser.add_argument('-o', '--output_file', type=str, default='predictions.json',
                        help='File to save output')
    parser.add_argument('-l', '--log_file', type=str, default='../logs/train_svm.log',
                        help='Log file to save grid search results')
    
    args = parser.parse_args()
    return args


def get_classifier(args):
    '''Returns a classifier given its name. The default settings for each model
    are the best settings found in the grid search''' 

    if args.classifier in ('nb', 'naive_bayes'):
        classifier = MultinomialNB(
            alpha=args.alpha,
            fit_prior=args.fit_prior
        )
    elif args.classifier in ('dt', 'decision_tree'): 
        classifier = DecisionTreeClassifier(
            criterion=args.criterion,
            splitter=args.splitter,
            max_depth=args.max_depth,
            max_features=None if args.max_features == 'None' else args.max_features,
            class_weight=None if args.class_weight == 'None' else args.class_weight
        )
    elif args.classifier in ('rf', 'random_forest'): 
        classifier = RandomForestClassifier(
            n_estimators=args.n_estimators,
            criterion=args.criterion,
            max_depth=args.max_depth,
            max_features=None if args.max_features == 'None' else args.max_features,
            class_weight=None if args.class_weight == 'None' else args.class_weight,
            n_jobs=-1
        )
    elif args.classifier in ('kn', 'k_neighbors'):
        classifier = KNeighborsClassifier(
            n_neighbors=args.n_neighbors,
            weights=args.weights,
            algorithm=args.algorithm,
            n_jobs=-1
        )
    elif args.classifier in ('svc', 'support_vector'):
        classifier = SVC(
            kernel=args.kernel,
            gamma=args.gamma,
            C=args.c,
            class_weight=None if args.class_weight == 'None' else args.class_weight
        )
    elif args.classifier in ('svcl_twitter', 'support_vector_linear_twitter',
                             'svcl_telegram', 'support_vector_linear_telegram'):
        classifier = LinearSVC(
            penalty=args.penalty,
            dual=args.dual,
            C=args.c,
            loss=args.loss,
            class_weight=None if args.class_weight == 'None' else args.class_weight
        )
    else:
        raise ValueError(f"Could not find model: {args.classifier}. Usage: python train.py [file_parameters] [model_name] [*model_parameters]")
    
    return classifier


def grid_search(classifier, param_grid, cv=3):
    '''Takes as input a classifier model or pipeline, a parameter grid in the form
    of a dictionary with parameters as keys and a list of values as value,
    and a value cv for cross validation.
    Make sure the keys start with 'cls__' when they are used in a Pipeline'''
    return GridSearchCV(classifier, param_grid, verbose=2, n_jobs=-1, cv=cv, refit=True, scoring='f1_macro')


if __name__ == '__main__':
    args = create_arg_parser()

    # Create the test and training sets with appropriate labels
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.validation_file)
    X_train, X_dev = X_train.split(), X_dev.split()

    # Get the classifier and set hyperparameters based on command line arguments
    classifier = get_classifier(args)
    
    # Combine classifier with a bag-of-words vectorizer
    vec = CountVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, ngram_range=(1,3))
    classifier = Pipeline([('vec', vec), ('cls', classifier)])

    # Fill in the param grid and uncomment the next line to do a grid search
    param_grid = {}
    #classifier = grid_search(classifier, param_grid)

    # Fit the specified classifier on the training data
    classifier.fit(X_train, Y_train)

    # Use the fitted classifier to predict classes on the test data
    Y_pred = classifier.predict(X_dev)

    # Save the predictions to an output file
    json.dump(list(Y_pred), open(args.output_file, "w"))

    # If a grid search was done, log the best parameters.
    try:
        logging.basicConfig(filename=args.log_file, encoding='utf-8', level=logging.DEBUG)
        logging.info(f'Best parameters:\n{classifier.best_params_}')
        logging.info(f'Parameter grid:\n{param_grid}')
    except:
        # classifier.best_params_ does not exist, program is finished
        quit()
