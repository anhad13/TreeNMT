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
from nltk.translate.bleu_score import sentence_bleu

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
        self.decoder_rnn= dy.LSTMBuilder(num_layers, hidden_dim*2, hidden_dim, model)
        self.pred=model.add_parameters((vocab_size, hidden_dim))
        self.hdim=hidden_dim
    def decode(self, context, trg, decorate=False):
        prev_out=dy.zeros((self.hdim))
        outputs=[]
        s=self.decoder_rnn.initial_state()
        for i in range(len(trg)):
            emb=dy.concatenate([context, prev_out])
            s=s.add_input(emb)
            prev_out=s.output()
            pre2=dy.parameter(self.pred)
            outputs.append(pre2*prev_out)
        return outputs
    def generate(self, context, trg, maxlen=100):
        prev_out=dy.zeros((self.hdim))
        outputs=[]
        s=self.decoder_rnn.initial_state()
        for i in range(maxlen):
            emb=dy.concatenate([context, prev_out])
            s=s.add_input(emb)
            prev_out=s.output()
            pre2=dy.parameter(self.pred)
            act_value=pre2*prev_out
            act_value=np.argmax(act_value.value())
            outputs.append(act_value)
            if act_value==1:
                return outputs
        return outputs



#glovePath="/Users/anhadmohananey/Downloads/glove/glove.6B.300d.txt"
target_type="arabic"
glovePath="/scratch/am8676/glove.840B.300d.txt"
if target_type=="german":
    source_train_file="data/enpr.s"
    destination_train_file="data/trainde.s"
    dev_source="data/dev_enp.s"
    dev_target="data/dev_de.s"
elif target_type=="chinese":
    source_train_file="data/train_ez.enp"
    destination_train_file="data/train_ez.zh"
    dev_source="data/dev_ez.enp"
    dev_target="data/dev_ez.zh"
elif target_type=="arabic":
    source_train_file="data/train_en_ar.enp"
    destination_train_file="data/train_en_ar.ar"
    dev_source="data/dev15_en_ar.enp"
    dev_target="data/dev15_en_ar.ar"
print("Building source vocab.")
source_vocab = glove2dict(glovePath)
c1=1
for k in source_vocab.keys():
    source_vocab[k]=c1
    c1+=1
data_type="bal"
print("Loading Source Data")
source_data= dataparser.read_tree_dataset(source_train_file, source_vocab, data_type=data_type)
print("Loading Target Data")
if target_type=="chinese":
    target_data, target_vocab = dataparser.read_plain_dataset_chinese(destination_train_file)
else:
    target_data, target_vocab = dataparser.read_plain_dataset(destination_train_file)
print("Loading Dev Source")
dev_source_data=dataparser.read_tree_dataset(dev_source, source_vocab, data_type=data_type)
print("Loading Dev Target")
if target_type=="chinese":
    dev_target_data = dataparser.read_plain_dataset_from_existing_vocab_chinese(dev_target, target_vocab)
else:
    dev_target_data = dataparser.read_plain_dataset_from_existing_vocab(dev_target, target_vocab)
model = dy.Model()
pdb.set_trace()
batch_size=64
eval_every=batch_size*400
trainer = dy.AdamTrainer(model, 0.001)
#trainer.set_clip_threshold(-1.0)
encoder = EncoderTreeLSTM(model, len(source_vocab)+1, 300, 300)
decoder = DecoderLSTM(model, len(target_vocab)+1, 300)
import time
dy.renew_cg()
eval_only=False
filename=open(target_type+data_type+str(eval_only)+str(int(time.time())), "w")
start_time=time.time()
losses=[]
num_epochs=10
filename_model=data_type+".baz_llr"
best_dev_loss=1000
#source_data=dev_source_data
#target_data=dev_target_data
if eval_only:
    model.populate(filename_model)
    actual_results=[]
    for i in range(len(dev_source_data)):
       out_enc,c=encoder.expr_for_tree(dev_source_data[i]) 
       outs=decoder.generate(out_enc, dev_target_data[i])
       actual_results.append(sentence_bleu([dev_target_data[i]],outs))
       #actual_results.append(outs)
       filename.write(str(outs))
       filename.write("----")
       filename.write(str(dev_target_data[i]))
       filename.write("BLEU: "+str(sentence_bleu([dev_target_data[i]],outs)))
       filename.flush()
       dy.renew_cg() 
    print(actual_results)
    filename.write("BLEU Score: "+str(np.average(actual_results))+"\n")
    filename.flush()
else:
    for epoch in range(num_epochs):
        print("Starting Epoch: "+str(epoch))
        filename.write("Starting Epoch: "+str(epoch))
        for j in range(len(source_data)):
            out_enc,c=encoder.expr_for_tree(source_data[j])
            outs=decoder.decode(out_enc, target_data[j])
            loss=[dy.pickneglogsoftmax(outs[k],target_data[j][k])for k in range(len(outs))]
            loss=dy.esum(loss)
            losses.append(loss)
            #print(j)
            if j%batch_size==0:
                net_loss=dy.esum(losses)/batch_size 
                net_loss.backward()
                #pdb.set_trace()
                try:
                    trainer.update()
                except: 
                    pdb.set_trace()
                    print("Possible gradient overflow, skipping training example(s).")
                difference=time.time()-start_time
                print(str(j)+"---"+str(difference)+":")
                filename.write(str(j)+"---"+str(difference)+":\n")
                filename.flush()
                losses=[]
                dy.renew_cg()
            if j>0 and j%eval_every==0:
                actual_results=[]
                for i in range(len(dev_source_data)):
                   out_enc,c=encoder.expr_for_tree(dev_source_data[i])
                   outs=decoder.generate(out_enc, dev_target_data[i])
                   val=bleu_evaluator().evaluate([dev_target_data[i]],[outs]).value()
                   if val==None:
		   	val=0.0
 		   actual_results.append(val)
		   #actual_results.append(bleu_evaluator().evaluate([dev_target_data[i]],[outs]).value())
                   #actual_results.append(sentence_bleu([dev_target_data[i]],outs))
                   #dy.renew_cg() 
                filename.write("BLEU Score: "+str(np.average(actual_results))+"\n")
                filename.flush()
                dy.renew_cg()
            if j>0 and j%10000==0:
                filename.write("Saving Model.Checkpointing.\n")
                filename.flush()
                #model.save(filename_model)
