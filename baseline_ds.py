from read_and_preprocess import preprocessed_train, preprocessed_dev, preprocessed_test
import datasets
from transformers import AutoTokenizer
import numpy as np
from datasets import load_metric

'''
This script contains functions and varaibles for the baseline model.

Dataset preparation:
    - get_mappings_dict(): Get the dictionary mapping string classes (e.g. 'ARG0') to int labels and its reverse.
    - create_word_sentlist(): Create dictionary containing the required data for generating desired huggingface dataset.
    - tokenize_and_align_labels(): Adapted from the example notebook. Solves label alignment after re-tokenization.
    
Evaluation:
    - compute_metrics(): Compute the overall percision, recall and f1.
    - reverse_label(): Map the int class labels back to strings.
    - class_results(): Compute the percision, recall and f1 for each class.

Variables:
    - label_dict, label_dict_rev: The mapping dictionary from string class to int class and its reverse.
    - tokenized_train, tokenized_dev, tokenized_test: The tokenized and ready-to-use datasets.
    
'''

metric = load_metric("seqeval")
task = "srl"
model_checkpoint = "bert-base-uncased" # bert-base-uncased for better percision, distilbert-base-uncased for faster run
batch_size = 16
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def get_mappings_dict(preprocessed_train,preprocessed_dev,preprocessed_test):
    '''
    This function creates dictionaries that maps string class labels to int (and its reverse).
    '''
    preps = preprocessed_train+preprocessed_dev+preprocessed_test
    arg_list=[]
    for sent in preps:
        for token in sent:
            arg_list.append(token['ARG'])
    arg_list = sorted(list(set(sorted(arg_list))))
    map_dict = {}
    for i in range(len(arg_list)):
        map_dict[arg_list[i]]=i
    #map_dict = dict(zip(set(sorted(arg_list_n)), range(len(set(arg_list_n)))))
    map_dict_reverse = {v: k for k, v in map_dict.items()}
    return map_dict, map_dict_reverse
    
label_dict, label_dict_rev = get_mappings_dict(preprocessed_train,preprocessed_dev,preprocessed_test)
label_list = sorted(label_dict_rev)

def create_word_sentlist(pre_list):
    '''
    Creating list of token lists with the ARG and V infomation.
    '''
    word_sentlist = []
    for sentence in pre_list:
        featdict = {}
        wordlist,args,pred = [],[],[]
        for token in sentence:
            wordlist.append(token['form'])
            if token['V'] == 'V':
                pred.append(token['form'])
            args.append(label_dict[token['ARG']])
        featdict['tokens'],featdict['srl_arg_tags'],featdict['pred'] = wordlist,args,pred
        word_sentlist.append(featdict)
    
    return word_sentlist

trainsent, devsent, testsent = create_word_sentlist(preprocessed_train), create_word_sentlist(preprocessed_dev), create_word_sentlist(preprocessed_test)
trainds, devds, testds = datasets.Dataset.from_list(trainsent), datasets.Dataset.from_list(devsent), datasets.Dataset.from_list(testsent)


def tokenize_and_align_labels(examples, label_all_tokens=True):
    '''
    This function solves label alignment after re-tokenization and creates sent+[SEP]+pred+[SEP] structure
    '''
    tokenized_inputs = tokenizer(examples["tokens"],examples['pred'], truncation=True, is_split_into_words=True)
    labels = []
    all_word_ids = []
    for i, label in enumerate(examples[f"{task}_arg_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if (word_idx is None):
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)
        all_word_ids.append(word_ids)
    
    
    tokenized_inputs['word_ids'] = all_word_ids
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_train = trainds.map(tokenize_and_align_labels, batched=True)
tokenized_dev = devds.map(tokenize_and_align_labels, batched=True)
tokenized_test = testds.map(tokenize_and_align_labels, batched=True)


def remove_ignored_index(predictions,labels):
    '''
    This functino removes the ignored labels for the special tokens (-100)
    '''
    actual_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    actual_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return actual_predictions, actual_labels
    
def compute_metrics(p):
    '''
    This function computes the overall metrics (percision, recall, f1, accuracy)
    '''
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2) # Most possible label

    # Remove ignored index (special tokens)
    actual_predictions, actual_labels = remove_ignored_index(predictions,labels)
    
    results = metric.compute(predictions=actual_predictions, references=actual_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def reverse_label(num_labels):
    '''
    This function reverse int class label back to strings.
    '''
    sentlist = []
    for sents in num_labels:
        wordlist = []
        for label in sents:
            wordlist.append(label_dict_rev[label])
        sentlist.append(wordlist)
    return sentlist


def class_results(predictions, labels):
    '''
    This function computes metrics for each class.
    '''
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=reverse_label(true_predictions), references=reverse_label(true_labels))
    return results


