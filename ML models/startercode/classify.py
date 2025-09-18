import argparse
import sys
from naivebayes import *
from logisticregression import *


def get_arguments():

    # Please do not change the naming of these command line options or delete them. You may add other options for other hyperparameters but please provide the default values you used, via the "default=" argument
    parser = argparse.ArgumentParser(description="Given a saved model and unlabelled text, neural networks classifier")
    parser.add_argument("-m", help="modelfile: the name/path of the model to load after training using train.py")
    parser.add_argument("-i",  help="inputfile: the name/path of the test file that has to be read one text per line")
    parser.add_argument("-o", help="outputfile: the name/path of the output file to be written")
    # TODO: add extra arguments here if you want to

    return parser.parse_args()


def main(args):

    if "nb" in args.m:
        model = NaiveBayes(model_file=args.m)
    elif "lr" in args.m:
        model = LogisticRegression(model_file=args.m)

    else:
        ## TODO Add any other models you wish to evaluate
        model = None
        
    trained_model = model.load_model()

    preds = model.classify(args.i, trained_model)
    
    ## Save the predictions: one label prediction per line
    with open(args.o, "w") as file:
        for pred in preds:
            file.write(pred+"\n")

# DON'T CHANGE THIS
if __name__ == "__main__":
    args = get_arguments()
    main(args)
    