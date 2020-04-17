
# coding: utf-8

# **Dependencies**

# In[20]:


from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import EarlyStopping

from Callbacks import GenerateText, LossAndError


# In[ ]:


parser = argparse.ArgumentParser()

parser.add_argument('steps', type=int, default=1,
                   help='Steps or Skips (Cuts down on the patterns)')

args = parser.parse_args()


# **Load Ascii Text and Create Map of unique chars to integer**

# In[1]:


inputFile = open("input.txt", 'r', encoding='utf-8').read()
vocab = sorted(list(set(inputFile)))
char_to_int = dict((c, i) for i, c in enumerate(vocab))
int_to_char = dict((i, c) for i, c in enumerate(vocab))
nChars = len(inputFile)
nVocab = len(vocab)

print("Total Chars: ", nChars)
print("Total Vocab: ", nVocab)


# **Prepare Dataset of input to output pairs encoded as integers**
# 
# if sequence length is 5
# - CHAPT --> E
# - HAPTE --> R

# In[11]:


def sequence(seqLength):
    dataX = []
    dataY = []
    for i in range(0, nChars - seqLength, args.steps):
        seqIn = inputFile[i:i + seqLength]
        seqOut = inputFile[i + seqLength]
        dataX.append([char_to_int[char] for char in seqIn])
        dataY.append(char_to_int[seqOut])
    return dataX, dataY


# In[2]:


def sequence_one_hot(seqLength):
    dataX = []
    dataY = []
    
    for i in range(0, nChars - seqLength, args.steps):
        dataX.append(text[i: i + seqLength])
        dataY.append(text[i + seqLength])
    print('nb sequences:', len(sentences))
    print('Text Length:', len(text))

    print('Vectorization...')
    x = np.zeros((len(dataX), seqLength, nVocab), dtype=np.bool)
    y = np.zeros((len(dataX), nVocab), dtype=np.bool)
    
    for i, sentence in enumerate(dataX):
        for t, char in enumerate(sentence):
            x[i, t, char_to_int[char]] = 1
        y[i, char_to_int[dataY[i]]] = 1
        
    return x, y


# In[23]:


seqLength = 40
dataX, dataY = sequence(seqLength)
print(dataX[0])
print(dataY[0])

nPatterns = len(dataX)
print(len(dataX[0]))
print("Total Patterns: ", nPatterns)


# Must transform list of input sequences into form [samples, timesteps, features]
# - Rescale the integers to the range 0-1 to make the patterns easier to learn
# - Convert the output patters (single char converted to int) into a one hot encoding

# In[24]:


#reshape X to be [samples, timesteps, features]
X = np.reshape(dataX, (nPatterns, seqLength, 1))
#normalize
X = X / float(nVocab)

#one hot encode output variable
Y = np_utils.to_categorical(dataY)
print(X.shape)
print(Y.shape)


# In[25]:


def get_model():
    model = keras.Sequential()
    model.add(keras.layers.LSTM(1024, 
                                input_shape = (X.shape[1], X.shape[2]),
                                return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(1024))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(Y.shape[1], activation='softmax'))
    return model


# In[ ]:


def get_eff_model():
    input_layer = tf.keras.layers.Input(input_shape = (X.shape[1], X.shape[2]))
    x = tf.keras.layers.LSTM(1024, return_sequences=True)(input_layer)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(1024)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(Y.shape[1], activation='softmax')(x)
        


# In[26]:


def train_model():
    model = get_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    
    #define the checkpoint
    filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    TimeStamp = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose = 1, save_best_only=True, mode='min')
    callback_list = [GenerateText(), TimeStamp]
    
    #train/fit the model
    model.fit(X, Y, epochs=15, batch_size = 128, callbacks=callback_list)
    model.save_weights("weights"+str(seqLength)+".h5") #weights
    model.save("model"+str(seqLength)+".h5")


# In[16]:


#https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/shakespeare_with_tpu_and_keras.ipynb#scrollTo=tU7M-EGGxR3E


# Define Checkpoint

# In[22]:


train = True
if(train):
    train_model()

