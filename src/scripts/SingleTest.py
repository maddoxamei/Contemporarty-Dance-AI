
# coding: utf-8

# **Dependancies**

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os, sys


# **Load and Pre-process Data**

# In[2]:


path = "../data/CSV Data/"

filename = path+"Andria_Excited_v1-BVH_"

pos_data = pd.read_csv(filename+"pos.csv")
rot_data = pd.read_csv(filename+"rot.csv")

#normalization force values from -1 to 1
rot_data = rot_data/180.0

#Add the root (hip) data for spacial movement
rot_data['Hips.pos.x'] = pos_data.pop('Hips.x')
rot_data['Hips.pos.y'] = pos_data.pop('Hips.y')
rot_data['Hips.pos.z'] = pos_data.pop('Hips.z')

#Making movement relative to an origin of 0,0,0 for consistancy within different dances
rot_data['Hips.pos.x'] = rot_data['Hips.pos.x'] + (-1*rot_data['Hips.pos.x'][0])
rot_data['Hips.pos.y'] = rot_data['Hips.pos.y'] + (-1*rot_data['Hips.pos.y'][0])
rot_data['Hips.pos.z'] = rot_data['Hips.pos.z'] + (-1*rot_data['Hips.pos.z'][0])

time = rot_data.pop('time') #maybe change to time change value instead? To indicate speed


# **Setup Dataset**

# In[4]:


data = rot_data.copy()
#target = rot_data.copy()
#data = data[:-1]
#target = target.drop([0])

#dataset = tf.data.Dataset.from_tensor_slices((data.values, target.values))

BATCH_SIZE = 1
N_TIMESTEPS = 10
N_ROWS = data.values.shape[0]
N_COLOMNS = data.values.shape[1]
print(N_ROWS, N_COLOMNS)

data = data.iloc[:].values
dataX = []
dataY = []

blah=True
for i in range(0, N_ROWS - N_TIMESTEPS, 1):
    seqIn = data[i: i+N_TIMESTEPS]
    seqOut = data[i+N_TIMESTEPS : i+N_TIMESTEPS+1]
    dataX.append(seqIn)
    dataY.append(seqOut)
#X shape [samples, timesteps, features]
#Y shape [samples, 1, features]
X, Y = np.array(dataX), np.array(dataY)
N_SAMPLES = len(dataX)

#reshape Y to be [samples, features]
Y = np.reshape(Y, (N_SAMPLES, N_COLOMNS))

print(N_SAMPLES)
print(X.shape)
print(Y.shape)
#reshape X to be [samples, timesteps, features]
#X = np.reshape(dataX, (N_SAMPLES, N_TIMESTEPS, N_COLOMNS))


testfile = "../data/Numpy Data"
xFile = os.path.join(testfile, "Andria_Excited_v1-BVH_X")
yFile = os.path.join(testfile, "Andria_Excited_v1-BVH_Y")
np.save(xFile, X)
np.save(yFile, Y)


# In[5]:


Y2 = np.load(yFile+".npy")


# **Setup RNN**

# In[73]:


def get_compiled_model(): #Deprecated
	model = keras.Sequential()
	model.add(keras.layers.InputLayer(input_shape = (N_TIMESTEPS, N_COLOMNS)))
	#model.add(keras.layers.InputLayer(input_shape = (N_TIMESTEPS, N_FEATURES), batch_size = BATCH_SIZE))
	model.add(keras.layers.LSTM(35, activation='relu', return_sequences = True))
	model.add(keras.layers.LSTM(35, activation='relu', return_sequences = True))
	model.add(keras.layers.Dense(165, activation='tanh'))

	model.compile(optimizer='adam',
	              loss='mse',
	              metrics=['accuracy'])
	return model


# In[7]:


def create_model(stateful):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(256, activation='relu', input_shape = (N_TIMESTEPS, N_COLOMNS), batch_size = BATCH_SIZE, return_sequences=True, stateful=stateful))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(256, activation='relu', stateful=stateful))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(N_COLOMNS, activation='tanh'))
    return model


# **Train RNN**

# In[9]:


def train_model():
    model = create_model(True)
    model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])
    print(model.summary())
    #define the checkpoint
    filepath="../logs/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose = 1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    #train/fit the model
    model.fit(X, Y, epochs=15, callbacks=callbacks_list)


# In[19]:


def trim_weights():
    minFile = ""
    minLoss = 100
    '''test = {"weights-improvement-19-1.2765.hdf5",
            "weights-improvement-20-1.8434.hdf5",
            "weights-improvement-8-1.1234.hdf5"}'''
    for file in os.listdir("./"):
    #for file in test:
        if file.endswith(".hdf5"):
            string = file.split('-')
            value = (float)(os.path.splitext(string[len(string)-1])[0])
            if(minLoss>value):
                minLoss=value
                minFile = file
    return minFile


# In[10]:


train = True
if(train):
    train_model()
else:
    pass

