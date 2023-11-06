python ../code/train_svm.py -t ../data/twitter/train.tsv -v ../data/twitter/test.tsv svcl_twitter
python ../code/evaluate.py -i ../code/predictions.json -v ../data/twitter/test.tsv 
