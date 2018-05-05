import dynet as dy
import numpy as np
import json
import argparse
import re
import codecs
import csv
import dataparser
from evaluator import BLEUEvaluator as bleu_evaluator
import pdb

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
        self.E = model.add_lookup_parameters((vocab_size , wdim), init='glorot')
    def expr_for_tree(self, tree, decorate=False):
        if(tree.isleaf()):
            E = self.E
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
        #self.pre_l=model.add_parameters((hidden_dim, hidden_dim))
        self.pred=model.add_parameters((vocab_size, hidden_dim))
        self.hdim=hidden_dim
    def decode(self, context, trg, decorate=False):
        prev_out=dy.zeros((self.hdim))
        outputs=[]
        for i in range(len(trg)):
            emb=dy.concatenate([context, prev_out])
            Ui,Uo,Uu = [dy.parameter(u) for u in self.US]
            Uf1= dy.parameter(self.UFS[0])
            bi,bo,bu,bf = [dy.parameter(b) for b in self.BS]
            #import pdb;pdb.set_trace()
            i = dy.logistic(bi+Ui*emb)
            o = dy.logistic(bi+Uo*emb)
            f = dy.logistic(bf+Uf1*emb)
            u = dy.tanh(bu+Uu*emb)
            c = dy.cmult(i,u) + dy.cmult(f,prev_out)
            h = dy.cmult(o,dy.tanh(c))
            if decorate: tree._e = h
            prev_out=c
            #pre1=dy.parameter(self.pre_l)
            pre2=dy.parameter(self.pred)
            #outputs.append(dy.log_softmax(pre2*h))
            outputs.append(pre2*h)
        return outputs

    def generate(self, context, trg, decorate=False, maxpossible=100):
        #greedy generation!
        prev_out=dy.zeros((self.hdim))
        outputs=[]
        for i in range(maxpossible):
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
            #pre1=dy.parameter(self.pre_l)
            pre2=dy.parameter(self.pred)
            out=dy.log_softmax(pre2*h)
            out=np.argmax(out)
            outputs.append(out)
            if out==1:
                print(outputs)
                print("-----")
                print(trg)
                return outputs
        print(outputs)
        print("---")
        print(trg)
        return outputs



glovePath="/Users/anhadmohananey/Downloads/glove/glove.6B.300d.txt"
#glovePath="/scratch/am8676/glove.840B.300d.txt"
source_train_file="data/enpr.s"
destination_train_file="data/trainde.s"
dev_source="data/dev_en.s"
dev_target="data/dev_de.s"
print("Building source vocab.")
source_vocab = glove2dict(glovePath)
c1=1
for k in source_vocab.keys():
    source_vocab[k]=c1
    c1+=1
print("Loading Source Data")
source_data= dataparser.read_tree_dataset(source_train_file, source_vocab)
print("Loading Target Data")
target_data, target_vocab = dataparser.read_plain_dataset(destination_train_file)
print("Loading Dev Source")
dev_source_data=dataparser.read_tree_dataset(dev_source, source_vocab)
print("Loading Dev Target")
dev_target_data = dataparser.read_plain_dataset_from_existing_vocab(dev_target, target_vocab)
model = dy.Model()
batch_size=10
eval_every=batch_size*50
trainer = dy.AdamTrainer(model, 0.001)
#trainer.set_clip_threshold(-1.0)
encoder = EncoderTreeLSTM(model, len(source_vocab)+1, 300, 300)
decoder = DecoderLSTM(model, len(target_vocab)+1, 300)
import time
dy.renew_cg()
filename=open("out.x.1", "w")
start_time=time.time()
losses=[]
num_epochs=5

#source_data=dev_source_data
#target_data=dev_target_data
for epoch in range(num_epochs):
    for j in range(len(source_data)):
        out_enc,c=encoder.expr_for_tree(source_data[j])
        outs=decoder.decode(out_enc, target_data[j])
        loss=[dy.pickneglogsoftmax(outs[k],target_data[j][k])for k in range(len(outs))]
        loss=dy.esum(loss)
        losses.append(loss)
        #print(j)
        if j%batch_size==0:
            net_loss=dy.esum(losses)/batch_size 
            net_loss.forward()
            net_loss.backward()
            #pdb.set_trace()
            try:
                trainer.update()
            except: 
                #pdb.set_trace()
                #model.reset_gradient()
                print("Possible gradient overflow, skipping training example(s).")
            difference=time.time()-start_time
            print(str(j)+"---"+str(difference)+":")
            filename.write(str(j)+"---"+str(difference)+":")
            filename.flush()
            losses=[]
            dy.renew_cg()
        if j>0 and j%eval_every==0:
            total_loss=0.0
            for i1 in range(len(dev_source_data)):
                out_enc,c=encoder.expr_for_tree(dev_source_data[i1])
                outs=decoder.decode(out_enc, dev_target_data[i1])
                if i1==0:
                    actual=">> "
                    #pdb.set_trace()
                    for k in outs:
                        #pdb.set_trace()
                        actual+= " "+str(np.argmax(k.value()))
                    filename.write(actual)
                    print(actual)
                    filename.write("----")
                    print("-----")
                    filename.write(str(dev_target_data[i1]))
                    print(str(dev_target_data[i1]))
                loss=[dy.pickneglogsoftmax(outs[k],dev_target_data[i1][k])for k in range(len(outs))]
                loss=dy.esum(loss)
                losses.append(loss)
                if i1%batch_size==0:
                    net_loss=dy.esum(losses) 
                    total_loss+=net_loss.value()              
                    losses=[]
                    dy.renew_cg()
            print("Dev Loss: "+str(total_loss/len(dev_source_data)))
            filename.write("Dev Loss: "+str(total_loss/len(dev_source_data)))
            filename.flush()
    #now calculate BLEU scores.
    #all_res=[]
    # for i in range(len(dev_source_data)):
    #     dy.renew_cg()
    #     out_enc,c =encoder.expr_for_tree(dev_source_data[i])
    #     outs=decoder.generate(out_enc, dev_target_data[i])
        #all_res.append(outs)
    #pdb.set_trace()
    #bleu_result=bleu_evaluator().evaluate(dev_target_data, all_res).score_str()
    #print("BLEU Score:"+bleu_result)
    #filename.write("BLEU Scrore:"+bleu_result)

