# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:28:47 2024

@author: DELL
"""


import tensorflow as tf
from keras.regularizers import l2
from keras.activations import sigmoid
from attention import AttentionWithContext 
tfk = tf.keras
tfkl = tf.keras.layers
tfkc = tf.keras.callbacks



class WeakGRU(tf.keras.Model):

    def __init__(self):
        super(WeakGRU, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')
        self.conv2 = tfkl.Conv2D(4, (3,3), padding='same', activation='relu')
        self.gru = tfkl.TimeDistributed(
            tfkl.Bidirectional(
                tfkl.LSTM(128, return_sequences=True, dropout=0.0, recurrent_dropout=0.0, kernel_regularizer=l2(0.005)),
                merge_mode='sum'
            )
        )
        self.dropout = tfkl.Dropout(0.1)
        self.pool1 = tfkl.MaxPool2D(pool_size=(1,2))
        self.att = AttentionWithContext()

        self.attention_weights = tfkl.Dense(1)

        self.classifier = tfkl.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        input_bag = inputs
        inst_conv1 = self.conv1(input_bag)
        inst_pool1 = self.pool1(inst_conv1)
        
        inst_conv2 = self.conv2(inst_pool1)
        inst_pool2 = self.pool1(inst_conv2)
        
        inst_pool2 = self.dropout(inst_pool2, training=training)

        inst_conv3 = self.gru(inst_pool2)

        inst_features = tf.squeeze(inst_conv3, axis=0)
        

        gated_attention = self.att(inst_features)

        gated_attention = tfkl.Flatten()(gated_attention)

        gated_attention = self.attention_weights(gated_attention)
        
        gated_attention = tf.transpose(gated_attention, perm=[1, 0])

        gated_attention = tfkl.Softmax()(gated_attention)

        inst_features = tfkl.Flatten()(inst_features)
      
        bag_features = tf.matmul(gated_attention, inst_features)
      
        bag_probability = self.classifier(bag_features)

        return bag_probability, gated_attention,bag_features
    

