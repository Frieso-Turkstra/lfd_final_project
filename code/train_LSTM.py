#!/usr/bin/env python

'''This script can be used to run LSTM experiments.
The current model is a single layer bi-directional model.
The number of layers can be changed with a command line argument (i.e: -nl 3).

To run the script and return a .json file of predictions on the test set for evaluation, 
assuming the train.txt, dev.txt and test.txt are in the same folder, use this command:

> python train_LSTM.py --train_file train.txt --dev_file dev.txt --test_file test.txt --embeddings glove_twitter100.json
'''

import json
import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
from tensorflow.keras.initializers import Constant
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
from utils import read_corpus

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", default='train.tsv', type=str,
                        help="Input file to learn from (default train.tsv)")
    parser.add_argument("-d", "--dev_file", type=str, default='dev.tsv',
                        help="Separate dev set to read in (default dev.tsv)")
    parser.add_argument("-t", "--test_file", type=str,
                        help="If added, use trained model to predict on test set")
    parser.add_argument("-e", "--embeddings", default='glove_twitter100.json', type=str,
                        help="Embedding file we are using (default glove_twitter100.json)")
    parser.add_argument("-te", "--trainable_embeddings", action="store_true",
                        help="Make pre-trained embeddings trainable")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.004,
                        help="Set the learning rate for the model (default 0.004)")
    parser.add_argument("-bs", "--batch_size", type=int, default=16,
                        help="Set the batch size for the model (default 16)")
    parser.add_argument("-nl", "--lstm_layers", type=int, default=1, choices=range(1,5),
                        help="Set the number of bidirectional LSTM layers for the model, from a range of 1 to 5 (default 1)")
    args = parser.parse_args()
    return args


def read_embeddings(embeddings_file):
    '''Read in word embeddings from file and save as numpy array'''
    embeddings = json.load(open(embeddings_file, 'r'))
    return {word: np.array(embeddings[word]) for word in embeddings}


def get_emb_matrix(voc, emb):
    '''Get embedding matrix given vocab and the embeddings'''
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    # Bit hacky, get embedding dimension from the word "the"
    embedding_dim = len(emb["the"])
    # Prepare embedding matrix to the correct size
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings that we can feed to embedding layer
    return embedding_matrix


def create_model(Y_train, emb_matrix, learning_rate, number_of_layers, trainable_embeddings):
    '''Create the Keras model to use'''
    # Define settings, you might want to create cmd line args for them
    loss_function = 'binary_crossentropy'
    optim = Adam(learning_rate=learning_rate)
    # Take embedding dim and size from emb_matrix
    embedding_dim = len(emb_matrix[0])
    num_tokens = len(emb_matrix)
    num_labels = len(set(Y_train)) - 1
    # Now build the model
    model = Sequential()
    if trainable_embeddings:
        model.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(emb_matrix),trainable=True))
    else:
        model.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(emb_matrix),trainable=False))
    for layer in range(number_of_layers)-1:
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(units=64)))
    model.add(Dense(units=num_labels, activation="sigmoid"))
    # Compile model using our settings, check for accuracy
    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])
    return model


def train_model(model, X_train, Y_train, X_dev, Y_dev, batch_size):
    '''Train the model here, batch size and epoch are command line arguments'''
    verbose = 1
    epochs = 50
    # Early stopping: stop training when there are three consecutive epochs without improving the validation loss
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    # Finally fit the model to our data
    model.fit(X_train, Y_train, verbose=verbose, epochs=epochs, callbacks=[callback], batch_size=batch_size, validation_data=(X_dev, Y_dev))
    # Print final accuracy for the model (clearer overview)
    loss, accuracy = model.evaluate(X_dev, Y_dev)
    print('Loss on own dev set: {0}'.format(round(loss, 3)))
    print('Accuracy on own dev set: {0}'.format(round(accuracy, 3)))
    return model


def main():
    '''Main function to train and test neural network given cmd line arguments'''
    args = create_arg_parser()

    # Read in the data and embeddings
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)
    embeddings = read_embeddings(args.embeddings)

    # Transform words to indices using a vectorizer
    vectorizer = TextVectorization(standardize=None, output_sequence_length=50)
    # Use train and dev to create vocab - could also do just train
    text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)
    vectorizer.adapt(text_ds)
    # Dictionary mapping words to idx
    voc = vectorizer.get_vocabulary()
    emb_matrix = get_emb_matrix(voc, embeddings)

    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back
    Y_dev_bin = encoder.fit_transform(Y_dev)

    # Create model
    model = create_model(Y_train, emb_matrix, args.learning_rate, args.lstm_layers, args.trainable_embeddings)

    # Transform input to vectorized input
    X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
    X_dev_vect = vectorizer(np.array([[s] for s in X_dev])).numpy()

    # Train the model
    model = train_model(model, X_train_vect, Y_train_bin, X_dev_vect, Y_dev_bin, args.batch_size)

    # Do predictions on specified test set and store them for evaluation
    if args.test_file:
        # Read in test set and vectorize
        X_test, Y_test = read_corpus(args.test_file)
        X_test_vect = vectorizer(np.array([[s] for s in X_test])).numpy()
        # Use the fitted model to predict classes on the test data
        Y_pred = model.predict(X_test_vect)
        # Save the predictions to an output file
        json.dump(list(Y_pred), open(args.output_file, "w"))

if __name__ == '__main__':
    main()
