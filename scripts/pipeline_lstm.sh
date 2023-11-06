python train_LSTM.py -i ../data/twitter/train.tsv -d ../data/twitter/dev.tsv -t ../data/twitter/test.tsv -em ../data/embed_matrix.npy
python evaluate.py -i predictions.json -v ../data/twitter/test.tsv 
