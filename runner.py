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
    def __init__(self, model, vocab_size, hidden_dim, num_layers=1):
        self.WS = [model.add_parameters((hidden_dim, hidden_dim*2)) for _ in "iou"]
        self.US = [model.add_parameters((hidden_dim, hidden_dim*2)) for _ in "iou"]
        self.UFS =[model.add_parameters((hidden_dim, hidden_dim*2)) for _ in "f"]
        self.BS = [model.add_parameters(hidden_dim) for _ in "iouf"]
        self.pre_l=model.add_parameters((hidden_dim, hidden_dim))
        self.pred=model.add_parameters((vocab_size, hidden_dim))
        self.hdim=hidden_dim
    def decode(self, context, trg, decorate=False):
        prev_out=dy.zeros((self.hdim))
        outputs=[]
        for i in range(len(trg)):
            #import pdb;pdb.set_trace()
            emb=dy.concatenate([context, prev_out])
            Ui,Uo,Uu = [dy.parameter(u) for u in self.US]
            Uf1= dy.parameter(self.UFS[0])
            bi,bo,bu,bf = [dy.parameter(b) for b in self.BS]
            #import pdb;pdb.set_trace()
            i = dy.logistic(bi+Ui*emb)
            o = dy.logistic(bi+Uo*emb)
            f = dy.logistic(bf+Uf1*emb)
            #print("hey")
            u = dy.tanh(bu+Uu*emb)
            c = dy.cmult(i,u) + dy.cmult(f,prev_out)
            h = dy.cmult(o,dy.tanh(c))
            if decorate: tree._e = h
            prev_out=c
            pre1=dy.parameter(self.pre_l)
            pre2=dy.parameter(self.pred)
            #h1=h
            #import pdb;pdb.set_trace()
            outputs.append(pre2*pre1*h)
        return outputs



#glovePath="/Users/anhadmohananey/Downloads/glove/glove.6B.300d.txt"
glovePath="/scratch/am8676/glove.840B.300d.txt"
source_train_file="data/enpr.s"
destination_train_file="data/trainde.s"
source_vocab = glove2dict(glovePath)
c=0
for k in source_vocab.keys():
    source_vocab[k]=c
    c+=1
print("Loading Source Data")
source_data= dataparser.read_tree_dataset(source_train_file, source_vocab)
print("Loading Target Data")
target_data, target_vocab = dataparser.read_plain_dataset(destination_train_file)
model = dy.Model()
batch_size=32
trainer = dy.AdamTrainer(model)
encoder = EncoderTreeLSTM(model, len(source_vocab), 100, 100)
decoder = DecoderLSTM(model, len(target_vocab), 100)
import time
dy.renew_cg()
start_time=time.time()
losses=[]
for j in range(len(source_data)):
    out_enc=encoder.expr_for_tree(source_data[j])
    outs=decoder.decode(out_enc[-1], target_data[j])
    loss=[dy.pickneglogsoftmax(outs[k],target_data[j][k])for k in range(len(outs))]
    loss=dy.esum(loss)
    losses.append(loss)
    print(j)
    if j%batch_size==0:
        net_loss=dy.esum(losses)/batch_size 
        net_loss.backward()
        trainer.update()
        difference=time.time()-start_time
        print(str(j)+"---"+str(difference)+":")
        losses=[]
        dy.renew_cg()

