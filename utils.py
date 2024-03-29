import numpy as np
import datasets
import baseline_ds
import advanced_ds


def read_and_process_test(testfile):
    """
    This function read and process the test file into list of sentences lists.
    """
    with open(testfile, 'r', encoding='utf8') as infile:
        fulllist, sentlist = [],[]
        for line in infile:
            line = line.strip()
            if (line != '\n') & (line.startswith("#") == False): # Not empty and not commented
                sentlist.append(line.split())
            if line.startswith("#") == True:
                sentlist = [i for i in sentlist if i] # Remove empty list
                fulllist.append(sentlist)
                sentlist = []
                continue
        res = [ele for ele in fulllist if ele != []] # remove empty list
        sent_token, sent_label = [],[]
        for sentences in res:
            tokenlist, labellist = [],[]
            for pairs in sentences:
                tokenlist.append(pairs[0])
                labellist.append(pairs[1])
            sent_token.append(tokenlist)
            sent_label.append(labellist)
    return sent_token,sent_label

def labels_to_tags(text,labels):
    '''
    This function convert string labels to int labels and filter out the predicates.
    '''
    sents=[]
    for t,l in zip(text,labels):
        lablist=[]
        for label,token in zip(l,t):
            if label == 'V':
                label = '_'
                pred = [token]
            lablist.append(baseline_ds.label_dict[label])
        dict = {'tokens':t, 'srl_arg_tags':lablist, 'pred':pred}
        sents.append(dict)
    return sents

def create_tokenized_ds(tag_list):
    '''
    This function creates tokenized datasets that are ready to use.
    '''
    caseds = datasets.Dataset.from_list(tag_list)
    tokenized_case = caseds.map(baseline_ds.tokenize_and_align_labels, batched=True)
    return tokenized_case

def create_tokenized_ds_adv(tag_list):
    '''
    This function creates tokenized datasets that are ready to use. For advanced model.
    '''
    caseds = datasets.Dataset.from_list(tag_list)
    tokenized_case = caseds.map(advanced_ds.tokenize_and_align_labels_adv, batched=True)
    return tokenized_case

def file_to_ds(testfile):
    '''
    This function warps up and create ready to use dataset from test .txt file
    '''
    test_t, test_l = read_and_process_test(testfile)
    tag_list = labels_to_tags(test_t, test_l)
    return create_tokenized_ds(tag_list), test_t

def file_to_ds_adv(testfile):
    '''
    This function warps up and create ready to use dataset from test .txt file For advanced model.
    '''
    test_t, test_l = read_and_process_test(testfile)
    tag_list = labels_to_tags(test_t, test_l)
    return create_tokenized_ds_adv(tag_list), test_t



def remove_and_reverse(predictions, labels):
    pred = np.argmax(predictions, axis=2)
    predlists, lablists = [],[]
    for i in range(len(labels)):
        preds, labs = [],[]
        for j in range(len(labels[i])):
            if labels[i][j] != -100:
                preds.append(baseline_ds.label_dict_rev[pred[i][j]])
                labs.append(baseline_ds.label_dict_rev[labels[i][j]])
        preds.pop()
        labs.pop()
        predlists.append(preds)
        lablists.append(labs)
    return predlists, lablists

def print_results(sents, pred, true):
    fail, succ = 0,0
    for i in range(len(sents)):
        if pred[i] == true[i]:
            print('------Sentence ',i,': Success------')
            succ+=1
        if pred[i] != true[i]:
            print('------Sentence ',i,': Failure------')
            fail+=1
        print('Sent: ',sents[i])
        print('Pred: ', pred[i])
        print('True: ', true[i])
        print()
    fail_rate = fail / len(sents)
    print('Failure rate: ', fail_rate)
    return fail_rate

def calculate_failrate(sents, pred, true):
    fail, succ = 0,0
    for i in range(len(sents)):
        if pred[i] == true[i]:
            succ+=1
        if pred[i] != true[i]:
            fail+=1
    fail_rate = fail / len(sents)
    print('Failure rate: ', fail_rate)
    return fail_rate