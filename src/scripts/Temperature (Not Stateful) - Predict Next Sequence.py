
# coding: utf-8

# **Dependencies**

# In[ ]:


from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import EarlyStopping

from Callbacks import GenerateText, LossAndError


# **Load Ascii Text and Create Map of unique chars to integer**

# In[ ]:


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
# - CHAPT --> HAPTE
# - HAPTE --> APTER

# In[11]:


def sequence(seqLength):
    dataX = []
    dataY = []
    for i in range(0, nChars - seqLength, 1):
        seqIn = inputFile[i:i + seqLength]
        seqOut = inputFile[i+1: i + seqLength+1]
        dataX.append([char_to_int[char] for char in seqIn])
        dataY.append([char_to_int[char] for char in seqOut])
    return dataX, dataY


# In[12]:


seqLength = 100
dataX, dataY = sequence(seqLength)
print(dataX[0])
print(dataY[0])

nPatterns = len(dataX)
print(len(dataX[0]))
print("Total Patterns: ", nPatterns)


# Must transform list of input sequences into form [samples, timesteps, features]
# - Rescale the integers to the range 0-1 to make the patterns easier to learn
# - Convert the output patters (single char converted to int) into a one hot encoding

# In[13]:


#reshape X to be [samples, timesteps, features]
X = np.reshape(dataX, (nPatterns, seqLength, 1))
#normalize
X = X / float(nVocab)

#one hot encode output variable
Y = np_utils.to_categorical(dataY)
print(X.shape)
print(Y.shape)


# In[14]:


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


# In[15]:


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

# In[17]:


train = True
if(train):
    train_model()

