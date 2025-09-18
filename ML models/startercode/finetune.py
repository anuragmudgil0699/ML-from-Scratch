import os
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import random
import torch
import argparse
import numpy as np

# Set seed (you can turn this off)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Tokenize the input
def tokenize_function(tokenizer, examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Evaluation code
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# Core training function -- you can modify this signature as needed if you want to add hyperparams
def do_train(model_name, train_set, eval_set, batch_size, epochs, lr, save_dir):

    # would need to change these for tasks other than binary sentiment classification
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # TODO: initialize tokenizer and collator
    
    # TODO: load model - you should pass id2label, label2id, and a value for num_labels

    # TODO: tokenize data - apply tokenize_function to train and eval sets
    # hint: use lambda x: tokenize_function(tokenizer, x) to pass tokenizer and data to map

    # TODO: initialize TrainingArguments
    # start withs some sensible defaults for hyperparams, then play around to improve performance!
    training_args = TrainingArguments()

    # TODO: initialize Trainer
    # should pass at least model, args, train_dataset, eval_dataset, tokenizer, data_collator, compute_metrics
    trainer = Trainer() 
    # TODO: run training

    ##### YOUR CODE ENDS HERE ######

    print("Training completed...")
    trainer.save_model(save_dir+"/final")
    return


# Core evaluation function
# model_dir is the on-disk model you want to evaluate
# out_file is the file you will write results to
# you shouldn't have to change this
def do_eval(eval_set, model_dir, batch_size, out_file):

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    tokenized_eval = eval_set.map(lambda x: tokenize_function(tokenizer, x),  batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.args.per_device_eval_batch_size=batch_size
    logits = trainer.predict(tokenized_eval)
    predictions = torch.argmax(torch.tensor(logits.predictions), dim=-1)
    labels = logits.label_ids
    score = logits.metrics["test_accuracy"]
    # write to output file
    for i in range(predictions.shape[0]):
        out_file.write(f"{predictions[i].item()} {labels[i].item()}\n")
    return score

# you shouldn't have to change the main function unless you change other function signatures.

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--train", action="store_true", help="train a model on the training data")
    parser.add_argument("--dataset", type=str, default="imdb", help="huggingface dataset to use")
    parser.add_argument("--eval", action="store_true", help="evaluate model on the test set")
    parser.add_argument("--model_name", type=str, default="distilbert/distilbert-base-cased", help="huggingface base model to use")
    parser.add_argument("--model_dir", type=str, default="./out")
    parser.add_argument("--eval_outfile", type=str, default="./distilbert-output.txt")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--small", action="store_true", help="use small dataset (for debugging)")

    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset(args.dataset)
    # it might take 30 min or so per epoch to train distillbert all 25k on my laptop; this takes about 70s
    # recommended: use a small dummy dataset to debug your code before training on the full dataset
    if args.small:
      print("Using small dataset")
      dataset["train"] = dataset["train"].shuffle(seed=42).select(range(4000))
      dataset["test"] = dataset["test"].shuffle(seed=42).select(range(1000))

    if args.train:
        do_train(args.model_name, dataset["train"], dataset["test"],
                 args.batch_size, args.num_epochs, args.learning_rate, args.model_dir)

  
    test_data = dataset["test"]
    # Evaluate the trained model on the original test dataset
    if args.eval:
        out_file = open(args.eval_outfile, "w")
        score = do_eval(test_data, args.model_dir+"/final", args.batch_size, out_file)
        print("Score: ", score)
        out_file.close()