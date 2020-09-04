
# coding: utf-8

# **Dependencies**

# In[2]:


from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import EarlyStopping




import datetime


# **General Format**
# 
# on_(train|test|predict)_begin(self, logs=None)
# 
# on_(train|test|predict)_end(self, logs=None)
# 
# on_(train|test|predict)_batch_begin(self, batch, logs=None)
# 
# on_(train|test|predict)_batch_end(self, batch, logs=None)
# 
# *For Training Only*
# 
# on_epoch_begin(self, epoch, logs=None)
# 
# on_epoch_end(self, epoch, logs=None)

# In[4]:


class TimeStamp(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

    def on_train_batch_end(self, batch, logs=None):
        print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

    def on_test_batch_begin(self, batch, logs=None):
        print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

    def on_test_batch_end(self, batch, logs=None):
        print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))


# In[5]:


class LossAndError(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))

    def on_test_batch_end(self, batch, logs=None):
        print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))

    def on_epoch_end(self, epoch, logs=None):
        print('The average loss for epoch {} is {:7.2f} and mean absolute error is {:7.2f}.'.format(epoch, logs['loss'], logs['mae']))


# **Text Generation**

# In[ ]:


def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)


# In[7]:


class GenerateText(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None): #changed from (epoch, _)
        textfile = open("sample-epoch-"+str(epoch)+".txt", "w")
        
        # Function invoked at end of each epoch. Prints generated text.
        print()
        print('----- Generating text after Epoch: %d' % epoch)

        start_index = random.randint(0, len(text) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            textfile.write('----- diversity:', diversity)
            print('----- diversity:', diversity)

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            textfile.write('----- Generating with seed: "' + sentence + '"')
            textfile.write(generated)
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                sentence = sentence[1:] + next_char
                
                textfile.write(next_char)
                textfile.close()
                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

