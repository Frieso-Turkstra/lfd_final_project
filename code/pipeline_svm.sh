python train_svm.py -t ../data/twitter/train.tsv -v ../data/twitter/test.tsv svcl_twitter
python evaluate.py -i predictions.json -v ../data/twitter/test.tsv 
