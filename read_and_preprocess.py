import copy

'''
This script reads the original conllu file and preprocess it into list of sentence lists.

The preprocessed dataset list have the following structure:
    Each sentence list is a list of token lists. 
    Each token list have 13 columns.
        If a sentence have 0 predicates, the column (list item) 12 and 13 (list[11] and list[12]) are set as None.
        If the sentence have multiple predicates, it will be duplicated to align the column number.
        If a sentence does not have record on line 11, it will be filled with '_'
'''



######################## READING AND PREPROCESSING ########################
def read_conll(conllfile):
    """
    This function read and process the conllu file into list of sentences lists.
    """
    with open(conllfile, 'r', encoding='utf8') as infile:
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
    return res

def preprocess_list(conlllist):
    """
    This function preprocess the lists into list of sentences list.
    Each sentence list is a list of token lists. Each token list have 13 columns.
        If a sentence have 0 predicates, the column (list item) 12 and 13 (list[11] and list[12]) are set as None.
        If the sentence have multiple predicates, it will be duplicated to align the column number.
        If a sentence does not have record on line 11, it will be filled with '_'
    """
    sentlist = []
    for sentence in conlllist:
        sents = [ [] for _ in range(50) ] # Initialize a large empty list for multiple predicate sentence    
        
        for x in range(len(sentence)): # replace 'for components in sentence' that brings duplicate removal error
            components = []
            for y in range(len(sentence[x])):
                components.append(str(sentence[x][y]))

            # First 11 lines
            for i in range(0,10):
                try:
                    tokendict = {"ID":components[0], "form":components[1], "lemma":components[2], "upos":components[3], "xpos":components[4], "feats":components[5], "head":components[6], 
                             "deprel":components[7], "deps":components[8], "misc":components[9], "pred":components[10]}
                except IndexError: # Wrong sentence in the dataset that have no column 11
                    tokendict['pred'] = '_'

            # If sentence have no predicate: assign the values '_'
            if len(components) <= 11: 
                tokendict['V'], tokendict['ARG'] ,tokendict['dup'] = '_','_','_'
                sents[0].append(tokendict)

            # Sentence have one or more predicate
            if len(components) > 11: 
                dup = len(components)-11 # Times for dpulication
                for k in range(0, dup):
                    tokendictk = copy.deepcopy(tokendict)
                    tokendictk['dup'] = k
                    ARGV = components[k+11]
                    # Following conditons change 'pred' (and ARG, V also) entry for duplicated sentence
                    if ARGV == 'V':
                        tokendictk['V'],tokendictk['ARG'] = 'V','_'
                        try:
                            tokendictk['pred'] = sentence[int(tokendictk['ID'])-1][10]
                        except IndexError:
                            #print('The following sentence contains error:',sentence)
                            continue
                    if (ARGV != 'V') & (ARGV != '_'):
                        tokendictk['ARG'],tokendictk['V'],tokendictk['pred'] = ARGV,'_','_'
                    if ARGV == '_':
                        tokendictk['V'],tokendictk['ARG'],tokendictk['pred'] = '_','_','_'
                    sents[k].append(tokendictk)


        res = [ele for ele in sents if ele != []] # remove empty list
        sentlist += res
    return sentlist

trainfile = 'data/en_ewt-up-train.conllu'
devfile = 'data/en_ewt-up-dev.conllu'
testfile = 'data/en_ewt-up-test.conllu'

trainlist, devlist, testlist = read_conll(trainfile), read_conll(devfile), read_conll(testfile)
preprocessed_train, preprocessed_dev, preprocessed_test = preprocess_list(trainlist), preprocess_list(devlist), preprocess_list(testlist)
