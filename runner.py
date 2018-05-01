import dynet as dy
import numpy as np
import json
import argparse
import re
import codecs
import csv
import dataparser


def glove2dict(src_filename):
    """GloVe Reader.

    Parameters
    ----------
    src_filename : str
        Full path to the GloVe file to be processed.

    Returns
    -------
    dict
        Mapping words to their GloVe vectors.

    """
    reader = csv.reader(open(src_filename), delimiter=' ', quoting=csv.QUOTE_NONE)
    return {line[0]: np.array(list(map(float, line[1: ]))) for line in reader}


class EncoderTreeLSTM(object):
    def __init__(self, model, vocab_size, wdim, hdim):
        self.WS = [model.add_parameters((hdim, wdim)) for _ in "iou"]
        self.US = [model.add_parameters((hdim, 2*hdim)) for _ in "iou"]
        self.UFS =[model.add_parameters((hdim, hdim)) for _ in "ff"]
        self.BS = [model.add_parameters(hdim) for _ in "iouf"]
        self.E = model.add_lookup_parameters((vocab_size , wdim))
    def expr_for_tree(self, tree, decorate=False):
        if(tree.isleaf()):
            E = dy.parameter(self.E)
            emb=E[tree.label]
            Wi,Wo,Wu   = [dy.parameter(w) for w in self.WS]
            bi,bo,bu,_ = [dy.parameter(b) for b in self.BS]
            #i = dy.logistic(dy.affine_transform([bi, Wi, emb]))
            #o = dy.logistic(dy.affine_transform([bo, Wo, emb]))
            #u = dy.tanh(    dy.affine_transform([bu, Wu, emb]))
            i = dy.logistic(bi+Wi*emb)
            o = dy.logistic(bo+Wo*emb)
            u = dy.tanh(    bu+Wu*emb)
            c = dy.cmult(i,u)
            h = dy.cmult(o,dy.tanh(c))
            if decorate: tree._e = h
            return h, c
        #now the 2 children case.
        e1, c1 = self.expr_for_tree(tree.children[0], decorate)
        e2, c2 = self.expr_for_tree(tree.children[1], decorate)
        Ui,Uo,Uu = [dy.parameter(u) for u in self.US]
        Uf1,Uf2 = [dy.parameter(u) for u in self.UFS]
        bi,bo,bu,bf = [dy.parameter(b) for b in self.BS]
        e = dy.concatenate([e1,e2])
        i = dy.logistic(bi+Ui*e)
        o = dy.logistic(bi+Uo*e)
        f1 = dy.logistic(bf+Uf1*e1)
        f2 = dy.logistic(bf+Uf2*e2)
        u = dy.tanh(     bu+Uu*e)
        c = dy.cmult(i,u) + dy.cmult(f1,c1) + dy.cmult(f2,c2)
        h = dy.cmult(o,dy.tanh(c))
        if decorate: tree._e = h
        return h, c

class DecoderLSTM(object):
    def __init__(self, model, vocab_size, hdim, num_layers=1):
        self.WS = [model.add_parameters((hdim, hdim*2)) for _ in "iou"]
        self.US = [model.add_parameters((hdim, hdim*2)) for _ in "iou"]
        self.UFS =[model.add_parameters((hdim, hdim*2)) for _ in "f"]
        self.BS = [model.add_parameters(hdim*2) for _ in "iouf"]
        self.pre_l=model.add_parameters((hidden_dim, hidden_dim))
        self.pred=model.add_parameters((hidden_dim,vocab_size))
    def decode(self, context, trg, decorate=False):
        prev_out=dy.zeros((hdim, hdim))
        outputs=[]
        for i in range(len(trg)):
            emb=dy.concatenate([context, prev_out])
            Ui,Uo,Uu = [dy.parameter(u) for u in self.US]
            Uf1= [dy.parameter(u) for u in self.UFS]
            bi,bo,bu,bf = [dy.parameter(b) for b in self.BS]
            i = dy.logistic(bi+Ui*emb)
            o = dy.logistic(bi+Uo*emb)
            f = dy.logistic(bf+Uf1*emb)
            u = dy.tanh(bu+Uu*emb)
            c = dy.cmult(i,u) + dy.cmult(f,c)
            h = dy.cmult(o,dy.tanh(c))
            if decorate: tree._e = h
            prev_out=c
            pre1=dy.parameter(self.pre_l)
            pre2=dy.parameter(self.pred)
            outputs.append(pre2*pre1*h)
        return outputs



glovePath="/Users/anhadmohananey/Downloads/glove/glove.6B.300d.txt"
source_train_file="data/enpr.s"
destination_train_file="data/trainde.s"
source_vocab = glove2dict(glovePath)
c=0
for k in source_vocab.keys():
    source_vocab[k]=c
    c+=1
source_data= dataparser.read_tree_dataset(source_train_file, source_vocab)
target_data, target_vocab = dataparser.read_plain_dataset(destination_train_file)

