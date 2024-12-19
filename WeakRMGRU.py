# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:09:03 2024

@author: DELL
"""

import time
import itertools
import numpy as np
import tensorflow as tf
#from nets import WSCNN, WSCNNLSTM, WeakRM, WeakRMLSTM
from utils import create_folder
from sklearn.model_selection import KFold,train_test_split
import pandas as pd
from WeakAtt import WeakGRU
from gensim.models import  Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,confusion_matrix


def circseq2ngram(k,s):
   
    seq_path="E:/m6A-circRNA/paper_data/all_file/m6a_pos.txt"
    with open(seq_path,"r") as fr:
        lines = fr.readlines()
    fr.close()
    words=np.zeros(shape=(1103,99)).astype(np.str_)
    word_path="E:/m6A-circRNA/paper_data/all_file/m6a_pos_word.txt"
    i=0
    for line in lines:
        #i=0
        j=0
        if line.startswith(">hsa") or len(line)<=1:
                continue
        else:
                line=line[:-1].upper()
                seq_len=len(line)
                for index in range(0,seq_len,s):
                    if index+k>=seq_len+1:
                        break
                    a=line[index:index+k]
                    words[i,j]=a
                    j=j+1
        i=i+1
    pd.DataFrame(words).to_csv("E:/m6A-circRNA/paper_data/all_file/m6a_pos_word.csv",index=False)
    
    with open(word_path,"w") as fw:
        for line in lines:
            if line.startswith(">hsa") or len(line)<=1:
                continue
            else:
                line=line[:-1].upper()
                seq_len=len(line)
                for index in range(0,seq_len,s):
                    if index+k>=seq_len+1:
                        break
                    fw.write("".join(line[index:index+k]))
                    fw.write(" ")
                fw.write("\n")
        fw.close()



def word2vec_train(vector_dim):
#    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",level=logging.INFO)
##    circseq2ngram(k,s)
#    circseq2ngram(3,1)    
    sentences=LineSentence("E:/m6A-circRNA/paper_data/all_file/m6a_pos_word.txt")
    model = Word2Vec(sentences, window=5, min_count=1, iter=30, size=vector_dim)
#    model.save("E:/m6A-circRNA/data/m6a_vec.h5")
#    model=load_model("E:/m6A-circRNA/data/m6a_vec.h5")
    dataset = pd.read_csv("E:/m6A-circRNA/paper_data/all_file/m6a_pos_word.csv",header=None)
    word = model.wv.index2word
    vector = model.wv.get_vector
  
    feature = np.zeros((1103,100,99))
    
    for i in range(0,1103):
        m=0 
        for j in range(0,99):
            char = dataset.iloc[i,j] 
#            index = word.index(char)
            feature[i,:,m] = vector(char)
            m=m+1

    dataset = pd.read_csv("E:/m6A-circRNA/paper_data/all_file/m6a_neg_word.csv")
   
    feature_n = np.zeros((1103,100,99))
    
    for i in range(0,1103):
        m=0 
        for j in range(0,99):
            char = dataset.iloc[i,j] 
            if(char!='NNN'):
                feature_n[i,:,m] = vector(char)
                m=m+1
            else:
                feature_n[i,:,m] = np.zeros(100)
                m=m+1
#            index = word.index(char)
            
    return feature,feature_n 

def get_bag(feature,label):
    bag = []
    number = np.unique(label)
    for i in range(0,len(number)):
        instance = feature[np.where(label==number[i])[0],:,:]
        bag.append(instance)
    return bag




def calc(y_true,y_pred):
    tn,fp,fn,tp = confusion_matrix(y_true,y_pred).ravel()
    f1 = f1_score(y_true,y_pred)
    acc = accuracy_score(y_true,y_pred)
    recall = recall_score(y_true,y_pred)
    pre = precision_score(y_true,y_pred)
    fz = tp*tn-fp*fn
    fm = (tp+fn)*(tp+fp)*(tn+fp)*(tn+fn)
    mcc = fz/pow(fm,0.5)
    auc = tf.keras.metrics.AUC()(y_true=y_true, y_pred=y_pred).numpy()
    aupr = tf.keras.metrics.AUC(curve='PR')(y_true=y_true, y_pred=y_pred).numpy()
    return f1,acc,recall,pre,mcc,auc,aupr

tfk = tf.keras
tfkl = tf.keras.layers
tfkc = tf.keras.callbacks

target_dir = 'E:/m6A-circRNA/data/test_data/'
create_folder(target_dir)
checkpoint_filepath = target_dir + 'WeakGRU' + '.weights.h5'

#feature_p,feature_n = load_data()
#    feature_p = feature_p.reshape((feature_p.shape[0],feature_p.shape[1],1))
#    feature_n = feature_n.reshape((feature_n.shape[0],feature_n.shape[1],1))
p_label = pd.read_csv("E:/m6A-circRNA/paper_data/all_file/index.csv").values
#n_label = pd.read_csv("E:/m6A-circRNA/ATG_n_label.csv",header = None).values
feature_p,feature_n = word2vec_train(100)
#feature_n = np.load("E:/m6A-circRNA/paper_data/feature_n.npy")
bag_p = get_bag(feature_p,p_label)
bag_n = get_bag(feature_n,p_label)
bag = bag_p+bag_n
y=np.append(np.ones((156,1),dtype=np.int32),np.zeros((156,1),dtype=np.int32))
y_prediction = y.flatten()
y_prob=np.ones((312,1),dtype=np.float32).flatten()
y=y.reshape(y.shape[0],1)
instance_len = 51

train_step_signature = [
    tf.TensorSpec(shape=(1, None, 100, 99), dtype=tf.float32),
    tf.TensorSpec(shape=(1, 1), dtype=tf.int32)
]


@tf.function(input_signature=train_step_signature)
def train_step(train_seq, train_label):
    with tf.GradientTape() as tape:
        output_probs, _ = model(train_seq, training=True)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true=train_label, y_pred=output_probs)
        total_loss = loss + tf.reduce_sum(model.losses)
        gradients = tape.gradient(total_loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_auROC(y_true=train_label, y_pred=output_probs)


@tf.function(input_signature=train_step_signature)
def valid_step(valid_seq, valid_label):
    inf_probs, _ = model(valid_seq, training=False)
    vloss = tf.keras.losses.BinaryCrossentropy()(y_true=valid_label, y_pred=inf_probs)
    valid_loss(vloss)
    valid_auROC(y_true=valid_label, y_pred=inf_probs)
auc = []
aupr = []
acc = []
mcc = []
precision = []
recall = []
f1 = []
bag_features =np.zeros((312,3200),dtype=np.float32)
kf = KFold(n_splits=5,shuffle=True)
for train_index,test_index in kf.split(bag):
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    for i in range(0,len(train_index)):
        train_X.append(bag[train_index[i]])
        train_Y.append(y[train_index[i]])
    for i in range(0,len(test_index)):
        test_X.append(bag[test_index[i]])
        test_Y.append(y[test_index[i]])
#    train_X = bag[train_index]
#    train_Y = y[train_index]
#    test_X = bag[test_index]
#    test_Y = y[test_index]
    t_X, v_X, t_Y, v_Y = train_test_split(bag, y, test_size=0.3,shuffle=True)
    data = lambda: itertools.zip_longest(train_X, train_Y)

    

    train_dataset = tf.data.Dataset.from_generator(lambda: itertools.zip_longest(t_X, t_Y),
                                               output_types=(tf.float32, tf.int32),
                                               output_shapes=(tf.TensorShape([None,100, 99]),
                                                              tf.TensorShape([None])))
    valid_dataset = tf.data.Dataset.from_generator(lambda: itertools.zip_longest(v_X, v_Y),
                                               output_types=(tf.float32, tf.int32),
                                               output_shapes=(tf.TensorShape([None, 100, 99]),
                                                              tf.TensorShape([None])))
    itest_dataset = tf.data.Dataset.from_generator(lambda: itertools.zip_longest(test_X, test_Y),
                                               output_types=(tf.float32, tf.int32),
                                               output_shapes=(tf.TensorShape([None, 100, 99]),
                                                              tf.TensorShape([None])))
#train_X = bag
#train_Y = y

#itest_dataset = train_dataset
    train_dataset = train_dataset.shuffle(100).batch(1)
    valid_dataset = valid_dataset.batch(1)
    itest_dataset = itest_dataset.batch(1)

    model_name = "WeakGRU"
    print('creating model')
    if isinstance(model_name, str):
       dispatcher = {
                     'WeakGRU': WeakGRU
                     
                    }
       try:
           model_funname = dispatcher[model_name]
       except KeyError:
           raise ValueError('invalid input')

    model = model_funname()


#lr = tf.keras.optimizers.schedules.ExponentialDecay(1e-4, decay_steps=len(train_label), decay_rate=0.96)
# opt = tf.keras.optimizers.Adam(learning_rate=lr)
    opt = tf.keras.optimizers.Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-5)

    train_loss = tf.keras.metrics.Mean()
    valid_loss = tf.keras.metrics.Mean()
    train_auROC = tf.keras.metrics.AUC()
    valid_auROC = tf.keras.metrics.AUC()



    train_step_signature = [
    tf.TensorSpec(shape=(1, None, 100, 99), dtype=tf.float32),
    tf.TensorSpec(shape=(1, 1), dtype=tf.int32)
    ]


    @tf.function(input_signature=train_step_signature)
    def train_step(train_seq, train_label):
        with tf.GradientTape() as tape:
           output_probs, _,_ = model(train_seq, training=True)
           loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true=train_label, y_pred=output_probs)
           total_loss = loss + tf.reduce_sum(model.losses)
           gradients = tape.gradient(total_loss, model.trainable_variables)
           opt.apply_gradients(zip(gradients, model.trainable_variables))
           train_loss(loss)
           train_auROC(y_true=train_label, y_pred=output_probs)


    @tf.function(input_signature=train_step_signature)
    def valid_step(valid_seq, valid_label):
       inf_probs, _,_ = model(valid_seq, training=False)
       vloss = tf.keras.losses.BinaryCrossentropy()(y_true=valid_label, y_pred=inf_probs)
       valid_loss(vloss)
       valid_auROC(y_true=valid_label, y_pred=inf_probs)
    EPOCHS = 20
    current_monitor = np.inf
    patient_count = 0

    for epoch in tf.range(1, EPOCHS+1):
       train_loss.reset_states()
       valid_loss.reset_states()

       train_auROC.reset_states()
       valid_auROC.reset_states()

       epoch_start_time = time.time()
       i = 0
       for tdata in train_dataset:
        #print(i)
           train_step(tdata[0], tdata[1])
           i=i+1
       print('Training of epoch {} finished! Time cost is {}s'.format(epoch, round(time.time() - epoch_start_time, 2)))

       valid_start_time = time.time()
       for vdata in valid_dataset:
           valid_step(vdata[0], vdata[1])


       new_valid_monitor = np.round(valid_loss.result().numpy(), 4)
       if new_valid_monitor < current_monitor:
           print('val_loss improved from {} to {}, saving model to {}'.
               format(str(current_monitor), str(new_valid_monitor), checkpoint_filepath))
           model.save_weights(checkpoint_filepath)
           current_monitor = new_valid_monitor
           patient_count = 0
       else:
           print('val_loss did not improved from {}'.format(str(current_monitor)))
           patient_count += 1

       if patient_count == 5:
           break

       template = "Epoch {}, Time Cost: {}s, TL: {}, TROC: {}, VL:{}, VROC: {}"
       print(template.format(epoch, str(round(time.time() - valid_start_time, 2)),
                          str(np.round(train_loss.result().numpy(), 4)),
                          str(np.round(train_auROC.result().numpy(), 4)),
                          str(np.round(valid_loss.result().numpy(), 4)),
                          str(np.round(valid_auROC.result().numpy(), 4)),
                          )
          )

    model.load_weights(checkpoint_filepath)
    predictions = []
    gateds = []
    bag_feature = []
    for tdata in itest_dataset:
         pred, gated_attention,bag_one_feature = model(tdata[0], training=False)
         predictions.append(pred.numpy())
         bag_feature.append(bag_one_feature.numpy().flatten())
         gateds.append(gated_attention.numpy())

    predictions = np.concatenate(predictions, axis=0).flatten()
    bag_features[test_index,:] = bag_feature
#    predictions = predictions.reshape((predictions.shape[0],1))
    y_prob[test_index] = predictions
    y_t = y_prediction[test_index]
    
    y_pred = predictions
    y_pred[np.array(predictions)>0.5] = 1
    y_pred[np.array(predictions)<0.5] = 0
    #y_pred = y_pred.flatten()
    f, ac, rec, pre, mc, au, aup = calc(y_t,y_pred)
    f1.append(f)
    acc.append(ac)
    recall.append(rec)
    precision.append(pre)
    mcc.append(mc)
    auc.append(au)
    aupr.append(aup)
d = np.vstack((f1,acc,recall,precision,mcc,auc,aupr))
pd.DataFrame(d).to_csv("E:/m6A-circRNA/paper_data/result/result_ndata.csv",index=False)
pd.DataFrame(bag_features).to_csv("E:/m6A-circRNA/paper_data/result/bagfeature_ndata.csv",index=False)
    
#    print('Test AUC: ', )
#    print('Test PRC: ', tf.keras.metrics.AUC(curve='PR')(y_true=y_t, y_pred=predictions).numpy())
    


#metric = np.zeros((1, 7))
#for train_index,test_index in kf.split(y):
#        train = []
#        y_train = []
#        for i in range(0,train_index.shape[0]):
#            train.append(bag[train_index[i]])
#            y_train.append(y[train_index[i]])
#        train_X, val_X, train_Y, val_Y = train_test_split(train, y_train, test_size=0.3,shuffle=True)
#        train_X = np.array(train_X)
#        train_Y = np.array(train_Y)
#        valid_X = np.array(val_X)
#        valid_Y = np.array(val_Y)
#    
