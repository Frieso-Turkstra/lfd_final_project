# lfd_final_project

## Dependencies

To install all the necessary dependencies, run the following command:

> pip install -r requirements.txt

## How to run the SVM:

To train and evaluate an SVM, simply run the 'pipeline_svm.sh' file from the code directory:

> python train_svm.py -t ../data/twitter/train.tsv -v ../data/twitter/test.tsv svcl_twitter  
> python evaluate.py -i predictions.json -v ../data/twitter/test.tsv 

The script takes a training file and a validation file as well as the name of a classifier. The specified classifier is fit on the training data and saves its predictions for the validation file in 'predictions.json'. This json file is used as input alongside the validation file for the evaluation script. The classification report and confusion matrix are stored in the 'evaluate.log' file in the logs directory.

To switch between datasets or models, simply swap 'twitter' for 'telegram' or vice versa.

## How to run the LSTM:
Similarly, to train and evaluate an LSTM, run the 'pipeline_LSTM.sh' from the code directory:

> python train_LSTM.py -i ../data/twitter/train.tsv -d ../data/twitter/dev.tsv -t ../data/twitter/test.tsv -em ../data/embed_matrix.npy
> python evaluate.py -i predictions.json -v ../data/twitter/test.tsv

The script takes a train, dev, and test file as parameters, as well as an embedding matrix, which can be created beforehand using the 'create_embed_matrix.py' script. 
The embedding matrix in the data folder was created using this script, based on the 100 dimension GloVe twitter embeddings, and the vocabulary from the train and dev sets of the Twitter corpus.
