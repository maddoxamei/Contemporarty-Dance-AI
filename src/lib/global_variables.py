import os
from .generator_dependencies import *

"""
 The values are based off of the models in 
        Recurrent Neural Networks for Modeling Motion Capture Data by Mir Khan, Heikki Huttunen, Olli Suominen and Atanas Gotchev
        Recurrent Network Models for Human Dynamics by Katerina Fragkiadaki, Sergey Levine, Panna Felsen, and Jitendra Malik
        Generative Choreography using Deep Learning by Luka Crnkovic-Friis and Louise Crnkovic-Friis
"""
""" 
===============================
======= Temp Variables ========
===============================   
"""
vertical_spacial_axis = 'Y' #The axis that is NOT being relativized (either None, X, Y, or Z)
shuffle_data = True #Shuffle windowed data during training
peep_hole = False
standardize_positions = True
relativize_positions = False
convensional_method = True #whether or not the rotational dataset should be standardized by convensional methods (val-mean)/std vs. val/180

""" 
======================================
======= Model Hyper-parameters =======
======================================

   ***** LSTM-RNN ***** 
"""
optimizer = 'RMSprop' # Maintain a moving (discounted) average of the square of gradients. Defaults are learning_rate=0.001, rho=0.9, momentum=0, epsilon=1e-07, centered=False
loss_function = 'mean_squared_error'
# The folowing initializers are applied to all hidden layers of the model before training 
weight_initializer = 'orthogonal' # Generates an orthogonal matrix with multiplicative factor equal to the gain. Defaults are gain=1.0, seed=None.
recurrent_initializer = 'glorot_normal' # Draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / (n_input_weight_units + n_output_weight_units)). Default is seed=None
bias_initializer = 'zeros' # Set biases to zero
layer_activation = 'tanh' #tanh(x) = sinh(x)/cosh(x) = ((exp(x) - exp(-x))/(exp(x) + exp(-x)))
recurrent_activation = 'sigmoid' #sigmoid(x) = 1 / (1 + exp(-x))     #hard_signmoid?????
output_activation = 'linear'
units = 1024 #number of nodes in each hidden layer of the network
""" 
    ***** MDN ***** 
"""
sigma_temp = 0.01
mixtures = 5 #number of mixture components
covariances = 5 #number of diagonal covariances

""" 
=========================================
======= Model Runtime-parameters ========
=========================================
"""
epochs = 200 #maximum number of times all training samples are fed into the model for training
#metrics=['accuracy','mean_absolute_error','mean_squared_logarithmic_error','poisson','root_mean_squared_error','hinge','kullback_leibler_divergence'] #the metrics to track during model training/evaluation
metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredLogarithmicError(), tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.Hinge(), tf.keras.losses.KLDivergence(), tf.keras.metrics.MeanSquaredError()]
stopping_patience = 20 #number of epochs with no improvement after which training will be stopped
temperature = 1
use_mdn = False
""" 
===============================
======= Data Variables ========
===============================   
"""
n_features = 165 #number of columns in the input dimension
frames = 500 #number of frames the model should generate

batch_size = 32 #number of samples trained before performing one step of gradient update for the loss function (default is stochastic gradient descent)
look_back = 50 #how many frames will be used as a history for prediction
offset = 1 #how many frames in the future is the prediction going to occur at
forecast = 1 #how many frames will be predicted
sample_increment = 1 #number of frames between each sample

training_split = 0.7 #the proportion of the data to use for training
validation_split = 0.2 #the proportion of the data to use for validating during the training phase at the end of each epoch
evaluation_split = 0.1 #(test_split) the proportion of the data for evaluating the model effectiveness after training completion

""" 
==============================
====== Path locations ========
==============================

    ***** Base *****
"""
_base_dir = os.path.dirname(os.getcwd())
_extras_dir = os.path.dirname(_base_dir)

csv_data_dir = os.path.join(_extras_dir, r"data/CSV/Raw") #directory to the csv representation of the dances
np_data_dir = os.path.join(_extras_dir, r"data/Numpy") #directory to the numpy representation of the dances
logs_dir = os.path.join(_base_dir, "logs") #general output directory
graphics_dir = os.path.join(_base_dir, "graphics")
hierarchy_dir = os.path.join(csv_data_dir, "hierarchy")
"""
    ***** Identifiers *****
"""
_model_identifier = "units-{}_timesteps-{}".format(units, look_back) #model-specific string for use in creating readily identifiable filenames
_data_identifier = "lb-{}_o-{}_f-{}_si-{}_sm-{}_rp-{}_sp-{}_va-{}".format(look_back, offset,forecast, sample_increment, convensional_method, relativize_positions, standardize_positions, vertical_spacial_axis) #data-specific string for use in creating readily identifiable filenames
_split_identifier = "ts-{}_vs-{}_es-{}".format(training_split, validation_split, evaluation_split) #split-specific string for use in creating readily identifiable filenames
_full_identifier = _model_identifier+"_"+_data_identifier+"_"+_split_identifier #string for use in creating readily identifiable filenames 
_checkpoint_extension = ".h5" #file type to save the model weights as, either tensorflow's default (.ckpt) or keras's default (.h5) 
_weights_filename = "weights_{}_".format(look_back)+_full_identifier+"_epoch-{epoch:02d}_loss-{loss:.5f}_val-loss-{val_loss:.5f}"+_checkpoint_extension #full filename to save training weights to
_model_type = 'LSTM-RNN'
if(use_mdn):
    _model_type += '_MDN'
    
_generated_bvh_filename = "{}-lb_".format(look_back)
if(relativize_positions):
    _generated_bvh_filename += "relativized_"
if(standardize_positions):
    _generated_bvh_filename += "standardized_"
_generated_bvh_filename += "method:{}_shuffled:{}".format(convensional_method, shuffle_data)
"""
    ***** Save Locations *****
"""
np_save_dir = os.path.join(np_data_dir, _model_identifier) #output directory for model-specific numpy content
logs_save_dir = os.path.join(logs_dir, _model_identifier) #output directory for model-specific non-numpy content
np_file_suffix = "_"+_data_identifier+"_ts-{}".format(training_split) #end half of the nu
"""
    ***** Save Files *****
"""
weights_file = os.path.join(logs_save_dir, _weights_filename)
architecture_file = os.path.join(logs_save_dir, "model_{}_architecture_".format(_model_type)+_model_identifier+".json")
model_file = os.path.join(logs_save_dir, "model_{}_full_".format(_model_type)+_full_identifier+".h5")

evaluation_filepath = os.path.join(np_save_dir, "_comprehensive_evaluation_"+_data_identifier+"_es-{}".format(evaluation_split))
history_train_file = os.path.join(logs_save_dir, "history_train_"+_full_identifier+".json")
history_eval_file = os.path.join(logs_save_dir, "history_eval_"+_full_identifier+".json")
standardization_json = os.path.join(logs_dir,'standardization_metrics.json')
label_json = os.path.join(logs_dir, "dance_labels.json")

generated_bvh_file = os.path.join(logs_save_dir, _generated_bvh_filename+".bvh")