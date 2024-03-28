from read_and_preprocess import preprocessed_train, preprocessed_dev, preprocessed_test
import datasets
from transformers import AutoTokenizer
import numpy as np
from datasets import load_metric

from baseline_ds import label_dict, label_dict_rev
task = "srl"

'''
This script contains functions and varaibles for the advanced model.

Dataset preparation:
    - augment_sentlist(): Create dictionary containing the predicate augmented data.
    - tokenize_and_align_labels_adv(): Adapted from the example notebook. Solves label alignment after re-tokenization.

Other functions and variables needed for the evaluation are the same as the baseline model.
    
'''

model_checkpoint = "bert-base-uncased" # bert-base-uncased for better percision, distilbert-base-uncased for faster run
batch_size = 16
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def augment_sentlist(pre_list):
    '''
    Mark the predicate with the augment strategy described in NegBERT.
    That is, adding a special token ([V]) immediately before the predicate.
    '''
    word_sentlist = []
    for sentence in pre_list:
        featdict = {}
        wordlist,args,pred = [],[],[]
        for token in sentence:
            if token['V'] == 'V':
                predicate = '[' + str('V') +'] ' + token['form']
                wordlist.append(predicate)
            else: wordlist.append(token['form'])
            args.append(label_dict[token['ARG']])
        featdict['tokens'],featdict['srl_arg_tags'] = wordlist,args
        word_sentlist.append(featdict)
    
    return word_sentlist

trainsent, devsent, testsent = augment_sentlist(preprocessed_train), augment_sentlist(preprocessed_dev), augment_sentlist(preprocessed_test)
trainds, devds, testds = datasets.Dataset.from_list(trainsent), datasets.Dataset.from_list(devsent), datasets.Dataset.from_list(testsent)


def tokenize_and_align_labels_adv(examples, label_all_tokens=True):
    '''
    This function solves label alignment after re-tokenization 
    and it does not consist the sent+[SEP]+pred+[SEP]
    '''
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
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
    
tokenized_train = trainds.map(tokenize_and_align_labels_adv, batched=True)
tokenized_dev = devds.map(tokenize_and_align_labels_adv, batched=True)
tokenized_test = testds.map(tokenize_and_align_labels_adv, batched=True)
