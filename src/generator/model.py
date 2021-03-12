from lib.generator_dependencies import *
#from .mdn import _MDN as MDN

def _LSTM_RNN(units, lb, feature_size, la, ra, wi, ri, bi, oa):
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
    input_layer = keras.layers.Input(shape=(lb, feature_size), name="inputs")
    lstm_layer1 = keras.layers.LSTM(units, 
                                activation = la,
                                recurrent_activation = ra,
                                kernel_initializer = wi,
                                recurrent_initializer = ri,
                                bias_initializer = bi,
                                return_sequences=True)(input_layer)
    lstm_layer2 = keras.layers.LSTM(units, 
                                activation = la,
                                recurrent_activation = ra,
                                kernel_initializer = wi,
                                recurrent_initializer = ri,
                                bias_initializer = bi,
                                return_sequences=True)(lstm_layer1)
    lstm_layer3 = keras.layers.LSTM(units, 
                                activation = la,
                                recurrent_activation = ra,
                                kernel_initializer = wi,
                                recurrent_initializer = ri,
                                bias_initializer = bi,
                                return_sequences=False)(lstm_layer2)
    output = keras.layers.Dense(feature_size, activation=oa)(lstm_layer3)
    model = keras.models.Model(inputs=input_layer, outputs=output)
    return model

def _LSTM_RNN_MDN(units, lb, feature_size, la, ra, wi, ri, bi, oa, mixtures):
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
    :param mixtures: the number of mixtures to create in the Mixture Density Layer
    :type int
    :return: the model's architecture
    :type keras.Model
    """
    input_layer = keras.layers.Input(shape=(lb, feature_size), name="inputs")
    lstm_layer1 = keras.layers.LSTM(units, 
                                activation = la,
                                recurrent_activation = ra,
                                kernel_initializer = wi,
                                recurrent_initializer = ri,
                                bias_initializer = bi,
                                return_sequences=True)(input_layer)
    lstm_layer2 = keras.layers.LSTM(units, 
                                activation = la,
                                recurrent_activation = ra,
                                kernel_initializer = wi,
                                recurrent_initializer = ri,
                                bias_initializer = bi,
                                return_sequences=True)(lstm_layer1)
    lstm_layer3 = keras.layers.LSTM(units, 
                                activation = la,
                                recurrent_activation = ra,
                                kernel_initializer = wi,
                                recurrent_initializer = ri,
                                bias_initializer = bi,
                                return_sequences=False)(lstm_layer2)
    output = MDN(feature_size, mixtures)(lstm_layer3)
    model = keras.models.Model(inputs=input_layer, outputs=output)
    return model

def establish_model(units, lb, feature_size, la, ra, wi, ri, bi, oa, mixtures, mdn_layer = False):
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
    :param mixtures:
    :type int
    :return: the model's architecture
    :type keras.Model
    """
    if(mdn_layer):
        return _LSTM_RNN_MDN(units, lb, feature_size, la, ra, wi, ri, bi, oa, mixtures)
    else:
        return _LSTM_RNN(units, lb, feature_size, la, ra, wi, ri, bi, oa)
