import json
import numpy as np
import argparse
from tensorflow.keras.layers import TextVectorization
from utils import read_corpus_lstm
import tensorflow as tf

'''This script can be used to generate an embedding matrix for use in LSTM experiments.
assuming the train.tsv and dev.tsv are in the same folder, use this command:

> python create_embed_matrix.py --input_file train.tsv --additional_file dev.tsv --embeddings glove_twitter100.json --output_file embed_matrix.npy
'''

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", default='train.tsv', type=str,
                        help="Train file to take vocabulary from (default train.tsv)")
    parser.add_argument("-si", "--additional_file", type=str,
                        help="Optional dev file to take additional vocabulary from")
    parser.add_argument("-e", "--embeddings", default='glove_twitter100.json', type=str,
                        help="Embedding file (default glove_twitter100.json)")
    parser.add_argument("-o", "--output_file", default='embed_matrix.npy', type=str,
                        help="File to store the embedding matrix (default embed_matrix.npy)")
    args = parser.parse_args()
    return args



def read_embeddings(embeddings_file):
    '''Read in word embeddings from file and save as numpy array'''
    embeddings = json.load(open(embeddings_file, 'r'))
    return {word: np.array(embeddings[word]) for word in embeddings}


def get_emb_matrix(voc, emb, output_file):
    '''Get embedding matrix given vocab and the embeddings and save it to the specified file.'''
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
    # Save the embedding matrix to the specified output file
    np.save(output_file, embedding_matrix)

def main():
    args = create_arg_parser()
    embeddings = read_embeddings(args.embeddings)
    X_train, Y_train = read_corpus_lstm(args.input_file)
    vectorizer = TextVectorization(standardize=None, output_sequence_length=50)
    if args.additional_file:
        X_dev, Y_dev = read_corpus_lstm(args.additional_file)
        text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)
    else:
        text_ds = tf.data.Dataset.from_tensor_slices(X_train)
    vectorizer.adapt(text_ds)
    # Dictionary mapping words to idx
    voc = vectorizer.get_vocabulary()
    emb_matrix = get_emb_matrix(voc, embeddings, args.output_file)


if __name__ == "__main__":
    main()