def read_corpus(corpus_file):
    '''Reads a string and returns a list of messages split by word
    and a list of corresponding labels.'''

    documents = []
    labels = []

    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.strip().split('\t')
            documents.append(tokens[0].split())
            labels.append(tokens[1])

    return documents, labels

def read_corpus_lstm(corpus_file):
    '''Reads a string and returns a list of messages in their entirety
    and a list of corresponding labels.'''

    documents = []
    labels = []

    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.strip().split('\t')
            documents.append(tokens[0])
            labels.append(tokens[1])

    return documents, labels
