import numpy as np
import json
import codecs
import re
import csv  

def read_tree_dataset(filename, vocab):
    with open(filename) as f:
        document=f.readlines()
    return [read_tree_line(x, vocab) for x in document]

def read_plain_dataset(filename):
    vocab={}
    c=2
    vocab["<s>"]=1 #to signify end of line.
    dataset=[]
    with open(filename) as f:
        for x in f.readlines():
            tmp=[]
            for v in x.lower().split():
                if v not in vocab:
                    vocab[v]=c
                    c+=1
                tmp.append(vocab[v])
            dataset.append(tmp+[1])
    return dataset, vocab

def read_plain_dataset_from_existing_vocab(filename, vocab):
    dataset=[]
    with open(filename) as f:
        for x in f.readlines():
            tmp=[]
            for v in x.lower().split():
                if v in vocab:
                    tmp.append(vocab[v])
                else:
                    tmp.append(0)
            dataset.append(tmp+[1])
    return dataset

def read_tree_line(line, vocab):
    to, tra=convert_binary_bracketing(line)
    return convert_to_tree(to, tra, vocab)


def convert_binary_bracketing(parse, lowercase=False):
    transitions = []
    tokens = []
    for word in parse.split(' '):
        if word[0] != "(":
            if word.strip() == ")":
                transitions.append(1)
            else:
                # Downcase all words to match GloVe.
                if lowercase:
                    tokens.append(word.lower())
                else:
                    tokens.append(word)
                transitions.append(0)
    return tokens, transitions


def convert_to_tree(tokens,transitions, vocab):
    h=[]
    tokens_c=0
    for i in range(len(transitions)):
        if transitions[i]==0:
            	if tokens[tokens_c].lower() in vocab:
                	val=vocab[tokens[tokens_c].lower()]
                else:
                	val=0
       		h.append(Tree(val, None))
        	tokens_c+=1
        elif transitions[i]==1:
            x1=h.pop()
            x2=h.pop()
            h.append(Tree(None, [x2, x1]))
        else:
            break
    return h.pop()

class Tree(object):
    def __init__(self, label, children=None):
        self.label = label
        self.children = children
    @staticmethod
    def from_sexpr(string):
        toks = iter(_tokenize_sexpr(string))
        assert next(toks) == "("
        return _within_bracket(toks)
    def __str__(self):
        if self.children is None: return self.label
        return "[%s %s]" % (self.label, " ".join([str(c) for c in self.children]))
    def isleaf(self): return self.children==None
    def leaves_iter(self):
        if self.isleaf():
            yield self
        else:
            for c in self.children:
                for l in c.leaves_iter(): yield l
    def leaves(self): return list(self.leaves_iter())
    def nonterms_iter(self):
        if not self.isleaf():
            yield self
            for c in self.children:
                for n in c.nonterms_iter(): yield n
    def nonterms(self): return list(self.nonterms_iter())
