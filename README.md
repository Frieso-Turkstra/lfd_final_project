# lfd_final_project

## How to run the SVM:

To train and evaluate an SVM, simply run the 'pipeline_svm.sh' file:

> python train_svm.py -t ../data/twitter/train.tsv -v ../data/twitter/test.tsv svcl_twitter  
> python evaluate.py -i predictions.json -v ../data/twitter/test.tsv 

The script takes a training file and a validation file as well as the name of a classifier. The specified classifier is fit on the training data and saves its predictions for the validation file in 'predictions.json'. This json file is used as input alongside the validation file for the evaluation script. The classification report and confusion matrix are stored in the 'evaluate.log' file in the logs directory.

To switch between datasets or models, simply swap 'twitter' for 'telegram' or vice versa.
