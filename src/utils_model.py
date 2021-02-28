import os, sys
import tensorflow as tf
from tensorflow import keras

def establish_model(units, lb, feature_size, la, ra, wi, ri, bi, oa):
    """ Establish the architecture (layers and how they are connected*) of the model with freshly initialized state for the weights. 
        There is NO compilation information.

    :param units: number of nodes in each hidden layer of the network
    :type int
    :param lb: how many frames will be used as a history for prediction
    :type int
    :param feature_size: the number of features in the input/output vector
    :type int
    :param la: activation function to use for all hidden layers
    :type keras.activations
    :param ra: activation function to use in the recurrent step for all hidden layers
    :type keras.activations
    :param wi: weight matrix to set initial values of all hidden layer kernels
    :type keras.initializers
    :param ri: weight matrix to set initial values of all hidden layer recurrent kernels
    :type keras.initializers
    :param bi: matrix to represent initial bias values in all hidden layers
    :type keras.initializers
    :param oa: keras.activations to use for the output layer
    :type keras.activations
    :return: the model's architecture
    :type keras.Model
    """
    model = keras.Sequential()
    model.add(keras.layers.LSTM(units, 
                                input_shape = (lb, feature_size), 
                                #batch_size = batch_size, 
                                activation = la,
                                recurrent_activation = ra,
                                kernel_initializer = wi,
                                recurrent_initializer = ri,
                                bias_initializer = bi,
                                return_sequences=True))
    model.add(keras.layers.LSTM(units, 
                                activation = la,
                                recurrent_activation = ra,
                                kernel_initializer = wi,
                                recurrent_initializer = ri,
                                bias_initializer = bi,
                                return_sequences=True))
    model.add(keras.layers.LSTM(units, 
                                activation = la,
                                recurrent_activation = ra,
                                kernel_initializer = wi,
                                recurrent_initializer = ri,
                                bias_initializer = bi,
                                return_sequences=False))
    model.add(keras.layers.Dense(feature_size, activation=oa))
    return model

def compile_model(model, optimizer, loss):
    """ Compile the given model so that it is ready for training and/or prediction/evaluation

    :param model: the model to compile
    :type keras.Model
    :return: the compiled model
    :type keras.Model
    """
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])#tf.keras.metrics.MeanSquaredError(),
    print(model.summary())
    return model

def save_architecture(model, out_path, identifier):
    """ Save the architecture (layers and how they are connected*). 
        Model can be created with a freshly initialized state for the weights and no compilation information from this savefile

    :param model: the model to save
    :type keras.Model
    :param out_path: directory to save the model to
    :type str
    :param identifier: unique string for creating readily identifiable filenames based off model specs
    :type str
    """
    json_config = model.to_jason()
    print(json_donfig)
    
def save_weights(model, out_path, save_filename):
    """ Save the model weights. Ideal for use during training to create checkpoints.
        Weights can be loaded into a model (ideally the original checkpointed model) to extract the desired weights/layers into the saved mode

    :param model: the model to save the weights from
    :type keras.Model
    :param out_path: directory to save the model to
    :type str
    :param save_filename: pre-fix name of the file to save
    :type str
    :param logs: dictionary containing current model specs
    :type dict
    """
    model.save_weights(os.path.join(out_path, save_filename))
    
def save_trained_model(model, out_path, identifier):
    """ Save the entire model. Model can be loaded and restart training right where you left off
        The following are saved:
            weight values
            Model's architecture
            Model's training configuration (what you pass to the .compile() method)
            Optimizer and its state, if any (this allows you to restart training)

    :param model: the model to save
    :type keras.Model
    :param out_path: directory to save the model to
    :type str
    :param identifier: unique string for creating readily identifiable filenames based off model specs
    :type str
    """
    model.save(os.path.join(out_path, "model-full_"+identifier+".h5"))
    
def load_architecture(file):
    """ Load the architecture (layers and how they are connected*). 
        Model can be created with a freshly initialized state for the weights.
        There is NO compilation information in this savefile.

    :param file: .json file which holds the model's architecture data
    :type str
    :return: the model's architecture
    :type keras.Model
    """
    return keras.models.model_from_json(file)

def load_trained_model(file):
    """ Load the pre-trained model. Compiled when loaded so training/prediction/evaluation can be restarted right where the model left off. 

    :param file: .h5 file which holds the model's information
    :type str
    :return: the compiled model
    :type keras.Model
    """
    return keras.models.load_model(file, compile=True)
