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
            return h, c, [h]
        #now the 2 children case.
        e1, c1, ea1 = self.expr_for_tree(tree.children[0], decorate)
        e2, c2, ea2 = self.expr_for_tree(tree.children[1], decorate)
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
        return h, c, [h]+ea1+ea2

class DecoderAttentionLSTM(object):
    def __init__(self, model, vocab_size, hidden_dim, num_layers=1, max_len=100):
        self.decoder_rnn= dy.LSTMBuilder(num_layers, hidden_dim*2, hidden_dim, model)
        self.pre_attend=model.add_parameters((max_len, max_len))
        self.v=model.add_parameters((hidden_dim, hidden_dim))
        self.u=model.add_parameters((1, hidden_dim))
        self.attender=model.add_parameters((hidden_dim, 2*hidden_dim))
        #self.attender=model.add_parameters((max_len, hidden_dim*(max_len+1)))
        self.pred=model.add_parameters((vocab_size, hidden_dim))
        self.hdim=hidden_dim
        self.max_len=max_len
    # def decode(self, h_a, trg, decorate=False):
    #     h_a+=([dy.zeros(self.hdim)]*(self.max_len-len(h_a)))#padding to make equal to maxlength
    #     h_al=dy.concatenate(h_a)
    #     h_ak=dy.concatenate(h_a, 1)
    #     prev_out=dy.zeros((self.hdim))
    #     outputs=[]
    #     s=self.decoder_rnn.initial_state()
    #     for i in range(len(trg)):
    #         attender=dy.parameter(self.attender)
    #         #pdb.set_trace()
    #         attention_weights=dy.softmax(attender*dy.concatenate([h_al, prev_out]))
    #         emb=dy.concatenate([h_ak*attention_weights, prev_out])
    #         s=s.add_input(emb)
    #         prev_out=s.output()
    #         pre2=dy.parameter(self.pred)
    #         outputs.append(pre2*prev_out)
    #     return outputs
    def decode(self, h_a, trg, decorate=False):
        h_a+=([dy.zeros(self.hdim)]*(self.max_len-len(h_a)))#padding to make equal to maxlength
        h_ak=dy.concatenate(h_a, 1)
        #pdb.set_trace()
        pre_attend=dy.parameter(self.pre_attend)
        context=h_ak*pre_attend
        prev_out=dy.zeros((self.hdim))
        outputs=[]
        s=self.decoder_rnn.initial_state()
        for i in range(len(trg)):
            attender=dy.parameter(self.attender)
            #pdb.set_trace()
            V=dy.parameter(self.v)
            tmp=dy.tanh(dy.colwise_add(context, V*prev_out))
            U=dy.parameter(self.u)
            attention_weights=dy.softmax(dy.transpose(U*tmp))
            #pdb.set_trace()
            emb=dy.concatenate([h_ak*attention_weights, prev_out])
            s=s.add_input(emb)
            prev_out=s.output()
            pre2=dy.parameter(self.pred)
            outputs.append(pre2*prev_out)
        return outputs
    def generate(self, context, trg, maxlen=150):
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
glovePath="/scratch/am8676/glove.840B.300d.txt"
source_train_file="data/enpr.s"
destination_train_file="data/trainde.s"
dev_source="data/dev_enp.s"
dev_target="data/dev_de.s"
print("Building source vocab.")
source_vocab = glove2dict(glovePath)
c1=1
for k in source_vocab.keys():
    source_vocab[k]=c1
    c1+=1
data_type="gt"
print("Loading Source Data")
source_data= dataparser.read_tree_dataset(source_train_file, source_vocab, data_type=data_type)
print("Loading Target Data")
target_data, target_vocab = dataparser.read_plain_dataset(destination_train_file)
print("Loading Dev Source")
dev_source_data=dataparser.read_tree_dataset(dev_source, source_vocab, data_type=data_type)
print("Loading Dev Target")
dev_target_data = dataparser.read_plain_dataset_from_existing_vocab(dev_target, target_vocab)
model = dy.Model()
batch_size=64
eval_every=batch_size*20
trainer = dy.AdamTrainer(model, 0.001)
#trainer.set_clip_threshold(-1.0)
encoder = EncoderTreeLSTM(model, len(source_vocab)+1, 300, 300)
decoder = DecoderAttentionLSTM(model, len(target_vocab)+1, 300)
import time
dy.renew_cg()
eval_only=False
filename=open("Av2out."+data_type+str(eval_only)+str(int(time.time())), "w")
start_time=time.time()
losses=[]
num_epochs=10
filename_model=data_type+".EDA"

#source_data=dev_source_data
#target_data=dev_target_data
if eval_only:
    model.populate(filename_model)
    actual_results=[]
    for i in range(len(dev_source_data)):
       out_enc,c, ha=encoder.expr_for_tree(dev_source_data[i]) 
       outs=decoder.generate(ha, dev_target_data[i])
       actual_results.append(sentence_bleu([dev_target_data[i]],outs))
       dy.renew_cg() 
    print(actual_results)
    filename.write("BLEU Score: "+str(np.average(actual_results))+"\n")
    filename.flush()
else:
    for epoch in range(num_epochs):
        print("Starting Epoch: "+str(epoch))
        filename.write("Starting Epoch: "+str(epoch))
        skipped=0
        for j in range(len(source_data)):
            out_enc,c, ha=encoder.expr_for_tree(source_data[j])
            if(len(ha)>100):
                skipped+=1
                print("Skipping > maxlen")
                filename.write("Skipped training ex. ")
                filename.flush()
                continue
            outs=decoder.decode(ha, target_data[j])
            loss=[dy.pickneglogsoftmax(outs[k],target_data[j][k])for k in range(len(outs))]
            loss=dy.esum(loss)
            losses.append(loss)
            if j%batch_size==0:
                net_loss=dy.esum(losses)/batch_size-skipped
                skipped=0 
                net_loss.backward()
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
                total_loss=0.0
                skipped_dev=0
                for i1 in range(len(dev_source_data)):
                    out_enc,c, ha=encoder.expr_for_tree(dev_source_data[i1])
                    if len(ha)>100:
                        skipped_dev+=1
                        continue
                    outs=decoder.decode(ha, dev_target_data[i1])
                    if i1==0:
                        actual=">> "
                        for k in outs:
                            actual+= " "+str(np.argmax(k.value()))
                        filename.write(actual+"\n")
                        print(actual)
                        filename.write("----\n")
                        print("-----")
                        filename.write(str(dev_target_data[i1])+"\n")
                        print(str(dev_target_data[i1]))
                    loss=[dy.pickneglogsoftmax(outs[k],dev_target_data[i1][k])for k in range(len(outs))]
                    loss=dy.esum(loss)
                    losses.append(loss)
                    if i1%batch_size==0:
                        net_loss=dy.esum(losses) 
                        total_loss+=net_loss.value()              
                        losses=[]
                        dy.renew_cg()
                print("Dev Loss: "+str(total_loss/(len(dev_source_data)-skipped_dev)))
                filename.write("Dev Loss: "+str(total_loss/(len(dev_source_data)-skipped_dev))+"\n")
                filename.flush()
            if j%10000==0:
        	model.save(filename_model)
        	filename.write("Saved Model.\n")
        	filename.flush()
