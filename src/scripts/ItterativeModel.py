
# coding: utf-8

# **Dependancies**

# In[2]:


from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys, random, argparse, time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
from keras.models import load_model
from keras.models import model_from_json


# **Determine Whether Train or Sample** 
# 
# Don't run in Jupyter Notebook

# In[ ]:


parser = argparse.ArgumentParser()

parser.add_argument('-train', action="store_true",
                   help='True: Train on dataset, False: Sample with trained model')

args = parser.parse_args()


# **Variables**

# In[4]:


csv_data_dir = "../data/CSV"
np_data_dir = "../data/Numpy"
save_dir = "../logs"
dances = []
BATCH_SIZE = 1
N_TIMESTEPS = 20
N_EPOCHS = 15


# ** Pull Names of Dance Data **

# In[3]:


def getFileNames():
    filenames = [f for f in os.listdir(csv_data_dir) if f.endswith('.csv')]
    for file in enumerate(filenames):
        filenames[file[0]] = file[1][:-7]
    return set(filenames)


# ** Pre-Process Data **

# In[4]:


def pre_process_data(filename):
    #filename = os.path.join(csv_data_dir)+filename
    
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
    data = rot_data.copy()

    return data


# ** Load Data and Separate Into Samples **

# In[5]:


def get_sample_data(filename):
    loadedX = os.path.join(np_data_dir, filename+"X-"+str(N_TIMESTEPS))
    loadedY = os.path.join(np_data_dir, filename+"Y-"+str(N_TIMESTEPS))
    
    if not (os.path.exists(loadedX+".npy") and os.path.exists(loadedY+".npy")):
        print("create")
        data = pre_process_data(os.path.join(csv_data_dir, filename))
        N_ROWS = data.values.shape[0]
        N_COLOMNS = data.values.shape[1]

        data = data.iloc[:].values
        dataX = []
        dataY = []

        for i in range(0, N_ROWS - N_TIMESTEPS, 1):
            seqIn = data[i: i+N_TIMESTEPS]
            seqOut = data[i+N_TIMESTEPS : i+N_TIMESTEPS+1]
            dataX.append(seqIn)
            dataY.append(seqOut)

        #X shape [samples, timesteps, features]
        #Y shape [samples, 1, features]
        X, Y = np.array(dataX), np.array(dataY)

        N_SAMPLES = len(dataX)
        Y = np.reshape(Y, (N_SAMPLES, N_COLOMNS))
        print("saving")
        np.save(loadedX, X)
        np.save(loadedY, Y)

    return np.load(loadedX+".npy"), np.load(loadedY+".npy")


# ** Set-Up Model **

# In[6]:


def create_model(N_COLOMNS, stateful):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(256, activation='relu', 
                                input_shape = (N_TIMESTEPS, N_COLOMNS), 
                                batch_size = BATCH_SIZE, 
                                return_sequences=True, 
                                stateful=stateful))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(256, activation='relu', stateful=stateful))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(N_COLOMNS, activation='tanh'))
    return model


# ** Train Model **

# In[4]:


def train_model():
    model = create_model(165, True)
    model.compile(optimizer='adam', loss='mse') #metrics=['accuracy']
    print(model.summary())
    
    dances = list(getFileNames())
    
    for i in range(N_EPOCHS):
        print(str(i)+"/"+str(N_EPOCHS))
        
        #define the checkpoint
        filepath = os.path.join(save_dir, "Itt-Weights-Improvement-"+str(i)+"-{loss:.4f}.h5")
        best_filepath = os.path.join(save_dir, "Itt-Weights-Best.h5")
        weight_improvement = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose = 1, save_best_only=True, mode='min')
        best_weights = keras.callbacks.ModelCheckpoint(best_filepath, monitor='loss', verbose = 1, save_best_only=True, mode='min')
        callbacks_list = [weight_improvement, best_weights]
        
        for dance in dances:
            print(str(i)+"/"+str(N_EPOCHS)+": on dance", dance)
            X, Y = get_sample_data(dance)
            #train/fit the model
            model.fit(X, Y, callbacks=callbacks_list)
        random.shuffle(dances)
    
    print("Done Training, Saving Model")
    savefile = os.path.join(save_dir, "Itt-Model-"+str(N_TIMESTEPS)+".h5")
    model.save(savefile) #arch+weight+optimizer state
    json_string = model.to_jason() #architecture
    
    savefile_weights = os.path.join(save_dir, "Itt-Model-Weights-"+str(N_TIMESTEPS)+".h5")
    model.save_weights(savefile_weights) #weights
    
    #model = load_model("model-Itt-"+str(N_TIMESTEPS)+".h5")
    #model = model_from_json(json_string)
    #model.load_weights("modelWeights-Itt-"+str(N_TIMESTEPS)+".h5")


# ** Sample Model **

# In[ ]:


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


# ** Run Script **

# In[ ]:


if(args.train):
    start_time = time.time()
    train_model()
    print("--- %s hours ---" % ((time.time() - start_time)/3600))
else:
    print("Will Sample in the Future")

