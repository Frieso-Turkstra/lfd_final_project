from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import LabelBinarizer
from utils import read_corpus
import argparse
import json

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--input_file", type=str, required=True,
                        help="Input file to make predictions on")
    parser.add_argument("-out", "--output_file", type=str, default='predictions.json',
                        help="Output file to store predictions in")
    
    models = parser.add_mutually_exclusive_group()
    models.add_argument("-twitter", "--twitter", action="store_true",
                        help="Use Twitter model for predictions")
    models.add_argument("-telegram", "--telegram", action="store_true",
                        help="Use Telegram model for predictions")
    models.required = True

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = create_arg_parser()

    # Read in data
    X_test, Y_test = read_corpus(args.input_file)

    # Tokenize the input data
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    tokens_test = tokenizer(X_test, padding=True, max_length=300,
    truncation=True, return_tensors="np").data

    # Encode test labels
    encoder = LabelBinarizer()
    Y_test_bin = encoder.fit_transform(Y_test)

    # Load in model from HugginFace
    if args.twitter:
        lm = "marieke93/roberta-offense-twitter"
    elif args.telegram:
        lm = "marieke93/roberta-offense-telegram"
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=1)

    # Get predictions
    predictions = model.predict(tokens_test)['logits']
    Y_pred = encoder.inverse_transform(predictions)

    # Save the predictions to an output file
    json.dump(list(Y_pred), open(args.output_file, "w"))