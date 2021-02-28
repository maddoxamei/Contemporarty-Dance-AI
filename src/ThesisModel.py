#!/usr/bin/env python
# coding: utf-8

# # Dependancies

# In[3]:


from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys, random, argparse, time, json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import utils
#No module named keras OR cannot import name 'np_utils' if tensorflow.keras
#from keras.utils import np_utils
#from keras.models import load_model
#from keras.models import model_from_json


# In[4]:


from importlib import reload


# # Variables

# **Hyper-perameters** 

# In[5]:


""" Parameters are based off of the 3 layer model in 
    Recurrent Neural Networks for Modeling Motion Capture Data 
    by Mir Khan, Heikki Huttunen, Olli Suominen and Atanas Gotchev
"""

optimizer = keras.optimizers.RMSprop(learning_rate=0.001) # Maintain a moving (discounted) average of the square of gradients
# The folowing initializers are applied to all hidden layers of the model before training 
weight_initializer = keras.initializers.Orthogonal(gain=1.0, seed=None) # Generates an orthogonal matrix with multiplicative factor equal to the gain
recurrent_initializer = tf.keras.initializers.GlorotNormal(seed=None) # Draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / (n_input_weight_units + n_output_weight_units))
bias_initializer = keras.initializers.Zeros() # Set biases to zero
layer_activation = 'tanh'
recurrent_activation = 'hard_sigmoid'
output_activation = 'linear'

batch_size = 32 #number of samples trained before performing one step of gradient update for the loss function (default is stochastic gradient descent)
look_back = 30 #how many frames will be used as a history for prediction
offset = 1 #how many frames in the future is the prediction going to occur at
forecast = 1 #how many frames will be predicted
sample_increment = 7 #number of frames between each sample
epochs = 30 #maximum number of times all training samples are fed into the model for training
units = 843 #number of nodes in each hidden layer of the network


# **Variables**

# In[6]:


n_features = 165 #number of columns in the input dimension
frames = 500 #number of frames the model should generate
training_split = 0.7 #the proportion of the data to use for training
validation_split = 0.2 #the proportion of the data to use for validating during the training phase at the end of each epoch
evaluation_split = 0.1 #(test_split) the proportion of the data for evaluating the model effectiveness after training completion

csv_data_dir = "/Akamai/MLDance/data/CSV/Raw" #directory to the csv representation of the dances
np_data_dir = "/Akamai/MLDance/data/Numpy" #directory to the numpy representation of the dances
logs_dir = "/Akamai/MLDance/logs" #general output directory

data_identifier = "lb-{}_o-{}_f-{}-si-{}".format(look_back, offset,forecast, sample_increment) #data-specific string for use in creating readily identifiable filenames
model_identifier = "units-{}_timesteps-{}".format(units, look_back) #model-specific string for use in creating readily identifiable filenames
split_identifier = "ts-{}_vs-{}_es-{}".format(training_split, validation_split, evaluation_split) #split-specific string for use in creating readily identifiable filenames
save_dir = os.path.join(logs_dir, model_identifier) #output directory for model-specific content


# In[7]:


out_file = open(os.path.join(save_dir, "outfile.txt"), "w")
#utils.write("test", out_file)
out_file.close()


# Run below block in Jupyter Notebook

# In[12]:


class Args():
    def __init__(self):
        self.train = False
        self.predict = False
        self.evaluate = False
args = Args()


# Do NOT run this in Jupyter Notebook

# In[13]:


parser = argparse.ArgumentParser()

#store_true: default is False, sets the value to True if the respective tag is called
#store_false: default is True, sets the value to False if the respective tag is called
parser.add_argument('--train', action="store_true",
                   help='Train on dataset')
parser.add_argument('--evaluate', action="store_true",
                   help='Run an evaluation on the trained model')
parser.add_argument('--predict', action="store_true",
                   help='Generate a dance using the trained model')

args = parser.parse_args()


# # Helper Functions

# **General**

# In[14]:


def create_dir(path):
    """ Create the cooresponding directory files for the given path if it does not yet exist

    :param path: proposed directory filepath
    :type str
    :return: created directory filepath
    :rtype: str
    """
    utils.create_dir(path)

def get_unique_dance_names():
    """ Aggregate the names of unique dances from the CSV data directory
    
    :return: the dance names where there are csv files for
    :rtype: list
    """
    return utils.get_unique_dance_names(csv_data_dir)


# ### Data Related
# 
# **Save/Load Functions**

# In[15]:


def get_processed_data(filename):
    """ Fetch the pre-procced data cooresponding to the given dance name

    :param filename: the name of the dance to grab the data from
    :type str
    :return: the pre-processed dance data
    :rtype: numpy.Array
    """
    csv_filename = os.path.join(csv_data_dir, filename)
    np_filename = os.path.join(np_data_dir, filename+"_"+data_identifier+"_ts-{}".format(training_split))
    return utils.get_processed_data(csv_filename, np_filename, training_split)
    
def save_generated_dance(generated_data, original_filename, save_filename):
    """ Save the generated dance to a csv file for bvh converstion later.

    :param generated_data: the name of the dance to grab the data from
    :type numpy.Array
    :param original_filename: dance name the generation seed is from
    :type str
    :param save_filename: the directory and filename to store the generated dance at
    :type str
    """
    hierarchy_file = os.path.join(csv_data_dir, "hierarchy/"+original_filename.split('_')[0]+"_hierarchy.csv")
    original_data = pd.read_csv(os.path.join(csv_data_dir, original_filename+"_rotations.csv"), nrows=0)
    c_headers = [c for c in original_data.columns if 'End' not in c ][1:]
    utils.save_generated_dance(generated_data, training_split, hierarchy_file, c_headers, save_filename)    


# **Sequence the Data (Separate Into Samples) Functions**

# In[16]:


def get_sample_data(filename):
    """ Fetch the pre-sequenced or sampled data from the given dance, or create/save it if it does not yet exist

    :param filename: the name of the dance to sample data from
    :type str
    :return: the collection of input X and target Y samples for the train, validation, and evaluation datasets
    :type tuple
    """    
    csv_filename = os.path.join(csv_data_dir, filename)
    np_filename = os.path.join(np_data_dir, filename+"_"+data_identifier+"_ts-{}".format(training_split))
    return utils.get_sample_data(csv_filename, np_filename, look_back, offset, forecast, sample_increment, training_split, validation_split)


# # Functions related to the Model

# **Set-Up Model**

# In[17]:


def establish_model(feature_size):
    """ Establish the architecture (layers and how they are connected*) of the model with freshly initialized state for the weights. 
        There is NO compilation information.

    :param feature_size: the number of features in the input/output vector
    :type int
    :return: the model's architecture
    :type keras.Model
    """
    return utils.establish_model(units, look_back, feature_size, layer_activation, recurrent_activation, weight_initializer, recurrent_initializer, bias_initializer, output_activation)

def compile_model(model):
    """ Compile the given model so that it is ready for training and/or prediction/evaluation

    :param model: the model to compile
    :type keras.Model
    :return: the compiled model
    :type keras.Model
    """
    return utils.compile_model(model, optimizer, 'mse')


# **Save and Load Helper Functions**

# In[18]:


def save_architecture(model, identifier):
    """ Save the architecture (layers and how they are connected*). 
        Model can be created with a freshly initialized state for the weights and no compilation information from this savefile

    :param model: the model to save
    :type keras.Model
    :param identifier: unique string for creating readily identifiable filenames based off model specs
    :type str
    """
    json_config = model.to_jason()
    print(json_donfig)
    
def save_weights(model, logs=None):
    """ Save the model weights. Ideal for use during training to create checkpoints.
        Weights can be loaded into a model (ideally the original checkpointed model) to extract the desired weights/layers into the saved mode

    :param model: the model to save the weights from
    :type keras.Model
    :param logs: dictionary containing current model specs
    :type dict
    """
    save_file = "weights_{}_".format(look_back)+model_identifier+"_loss-{:.2f}_acc-{:.2f}.h5".format(logs["loss"], logs["accuracy"])
    utils.save_weights(model, save_dir, save_file)
    
def save_trained_model(model, identifier):
    """ Save the entire model. Model can be loaded and restart training right where you left off
        The following are saved:
            weight values
            Model's architecture
            Model's training configuration (what you pass to the .compile() method)
            Optimizer and its state, if any (this allows you to restart training)

    :param model: the model to save
    :type keras.Model
    :param identifier: unique string for creating readily identifiable filenames based off model specs
    :type str
    """
    utils.save_trained_model(model, save_dir, identifier)
    
def load_architecture(file):
    """ Load the architecture (layers and how they are connected*). 
        Model can be created with a freshly initialized state for the weights.
        There is NO compilation information in this savefile.

    :param file: .json file which holds the model's architecture data
    :type str
    :return: the model's architecture
    :type keras.Model
    """
    return utils.load_architecture(file)

def load_trained_model(file):
    """ Load the pre-trained model. Compiled when loaded so training/prediction/evaluation can be restarted right where the model left off. 

    :param file: .h5 file which holds the model's information
    :type str
    :return: the compiled model
    :type keras.Model
    """
    return utils.load_trained_model(file)


# # Train Model

# In[19]:


class CustomCallback(keras.callbacks.Callback):
    """ A class to create custom callback options. This overrides a set of methods called at various stages of training, testing, and predicting. 
        Callbacks are useful to get a view on internal states and statistics of the model during training.
            Callback list can be passed for .fit(), .evaluate(), and .predict() methods
            
        keys = list(logs.keys())
    """
    def __init__(self):
        super(CustomCallback, self).__init__()
        self.start_time = None
        
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print("\tTraining, Beginning:", self.start_time)

    def on_train_end(self, logs=None):
        print("Training Complete ", "--- %s hours ---" % ((time.time() - start_time)/3600))

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_test_begin(self, logs=None):
        self.start_time = time.time()
        print("\tEvaluation, Beginning:", self.start_time)

    def on_test_end(self, logs=None):
        print("Evaluation Complete ", "--- %s minutes ---" % ((time.time() - start_time)/60))

    def on_predict_begin(self, logs=None):
        self.start_time = time.time()
        print("\tPredicting, Beginning:", self.start_time)

    def on_predict_end(self, logs=None):
        print("Predicting Complete ", "--- %s minutes ---" % ((time.time() - start_time)/60))

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        pass


# In[37]:


def train_model(model):
    """ Trains the model with the dance data.
        The History object's History.history attribute is a record of training loss values and metrics values at successive epochs, 
            as well as cooresponding validation values (if applicable).  

    :param model: the model to train
    :type keras.Model
    :return: the class containing the training metric information and the trained model
    :type tuple
    """
    
    dances = get_unique_dance_names()
    checkpoint_filename = "weights_{}_".format(look_back)+model_identifier+"_epoch-{epoch:02d}_loss-{loss:.2f}_acc-{accuracy:.2f}_val-loss-{val_loss:.2f}_val-acc-{val_accuracy:.2f}.h5"
    checkpoint_filepath = os.path.join(save_dir, checkpoint_filename)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', mode='auto', save_best_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=2, mode='auto', baseline=None, restore_best_weights=True)
    callbacks_list = [checkpoint, early_stopping, CustomCallback()]

    comprehensive_train_X = np.array([])
    comprehensive_train_Y = np.array([])
    comprehensive_validate_X = np.array([])
    comprehensive_validate_Y = np.array([])
    comprehensive_evaluation_X = np.array([])
    comprehensive_evaluation_Y = np.array([])
    
    print("Fetching and Agregating Training Data ...")
    for dance in utils.progressbar(dances, "Progress: "):
        train_X, train_Y, validate_X, validate_Y, evaluation_X, evaluation_Y = get_sample_data(dance)
        if(len(comprehensive_train_X)==0):
            comprehensive_train_X = train_X
            comprehensive_train_Y = train_Y
            comprehensive_validate_X = validate_X
            comprehensive_validate_Y = validate_Y
            comprehensive_evaluation_X = evaluation_X
            comprehensive_evaluation_Y = evaluation_Y
        else:
            comprehensive_train_X = np.vstack((comprehensive_train_X,train_X))
            comprehensive_train_Y = np.vstack((comprehensive_train_Y,train_Y))
            comprehensive_validate_X = np.vstack((comprehensive_validate_X,validate_X))
            comprehensive_validate_Y = np.vstack((comprehensive_validate_Y,validate_Y))
            comprehensive_evaluation_X = np.vstack((comprehensive_evaluation_X,evaluation_X))
            comprehensive_evaluation_Y = np.vstack((comprehensive_evaluation_Y,evaluation_Y))
      
    start_time = time.time()
    history = model.fit(comprehensive_train_X, comprehensive_train_Y, 
                        batch_size = batch_size, 
                        callbacks=callbacks_list, 
                        validation_data= (comprehensive_validate_X, comprehensive_validate_Y),
                        epochs=epochs, 
                        verbose=1)
    
    save_trained_model(model, model_identifier)
    evaluation_save = os.path.join(np_data_dir, "comprehensive_evaluation_"+data_identifier+"_es-{}".format(evaluation_split))
    np.save(evaluation_save+"_X", comprehensive_evaluation_X)
    np.save(evaluation_save+"_Y", comprehensive_evaluation_Y)
    with open(os.path.join(save_dir, "history_train_"+model_identifier+"_"+data_identifier+"_"+split_identifier+".json"), "w") as history_file:  
        json.dump(pd.DataFrame.from_dict(history.history).to_dict(), history_file) 
    print("Saved metric history to json file")
    return history, model, comprehensive_evaluation_X, comprehensive_evaluation_Y


# # Sample/Run Model (Make Predictions)

# In[21]:


def benchmark(model, n_frames, random_frame=False):
    """ Generate a dance sequence with the given model

    :param model: the model to use for prediction 
    :type keras.Model
    :param n_frames: the number of frames the model should generate
    :type int
    """
    #select random dance for seed
    dances = get_unique_dance_names()
    seed_dance_index = random.randint(0, len(dances) - 1)
    dance = get_processed_data(dances[seed_dance_index])
    seed = dance[:look_back]
    if random_frame:
        #select random frame(s) for seed
        seed_frame_index = random.randint(0, len(dance) - (look_back+1))
        seed = dance[seed_frame_index:seed_frame_index+look_back]
    
    print("Generating dance with seed from", dances[seed_dance_index])
    #for diversity in [0.2, 0.5, 1.0, 1.2]:
    for diversity in [1.0]:
        start_time = time.time()
        generated = seed
        for i in utils.progressbar(range(n_frames),"{} Progress: ".format(diversity)):
            preds = model.predict(np.array([generated[-look_back:]]), verbose=0)[0]
            generated = np.vstack((generated, preds))
        filename = os.path.join(save_dir, "generated_dance_{}-frames_{}-diversity".format(n_frames, diversity))
        save_generated_dance(generated, dances[seed_dance_index], filename)
        
        print("\tSaved to", filename)


# # Run Script

# In[1]:


def main():
    """ Driver function to control what is run and when if this is the main python script being ran.
        As the project was developed in a jupyter notebook, everything is self-contained in the main file.
        Any expansion, however, would be able to use the predefined classes and functions for whatever purpose without running anything.
    """
    reload(utils)
    save_location = utils.create_dir(save_dir)
    history, model, eval_X, eval_Y = None, None, None, None
    if(not args.train and not args.evaluate and not args.predict):
        print("Type -h and get a list of possible tasks. You may select multiple.")
    else:
        if(args.train):
            model = establish_model(n_features)
            model = compile_model(model)
            history, model, eval_X, eval_Y = train_model(model)
        if(args.evaluate):
            if(not model):
                #loads the most recent saved model
                filename = [f for f in os.listdir(save_dir) if "model" in f][-1]
                model = load_trained_model(os.path.join(save_location, filename))
                print(model.summary())
                eval_X = np.load(os.path.join(np_data_dir, "comprehensive_evaluation_"+data_identifier+"_es-{}".format(evaluation_split)+"_X.npy"))
                eval_Y = np.load(os.path.join(np_data_dir, "comprehensive_evaluation_"+data_identifier+"_es-{}".format(evaluation_split)+"_Y.npy"))
            history = model.evaluate(eval_X, eval_Y, callbacks=[], verbose=1)
            with open(os.path.join(save_dir, "history_eval_"+model_identifier+"_"+data_identifier+"_"+split_identifier+".json"), "w") as history_file:  
                json.dump(pd.DataFrame.from_dict(history.history).to_dict(), history_file) 
        print("Saved metric history to json file")
        if(args.predict):
            if(not model):
                #loads the most recent saved model
                filename = [f for f in os.listdir(save_dir) if "model" in f][-1]
                model = load_trained_model(os.path.join(save_location, filename))
                print(model.summary())
            benchmark(model, frames)
        
if __name__ == "__main__":
    main()


# In[ ]:




