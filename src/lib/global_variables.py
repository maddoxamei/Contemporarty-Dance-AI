import os

""" Parameters are based off of the 3 layer model in 
    Recurrent Neural Networks for Modeling Motion Capture Data 
    by Mir Khan, Heikki Huttunen, Olli Suominen and Atanas Gotchev
"""

optimizer = 'RMSprop' # Maintain a moving (discounted) average of the square of gradients. Defaults are learning_rate=0.001, rho=0.9, momentum=0, epsilon=1e-07, centered=False
loss_function = 'mse'
# The folowing initializers are applied to all hidden layers of the model before training 
weight_initializer = 'orthogonal' # Generates an orthogonal matrix with multiplicative factor equal to the gain. Defaults are gain=1.0, seed=None.
recurrent_initializer = 'glorot_normal' # Draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / (n_input_weight_units + n_output_weight_units)). Default is seed=None
bias_initializer = 'zeros' # Set biases to zero
layer_activation = 'tanh' #tanh(x) = sinh(x)/cosh(x) = ((exp(x) - exp(-x))/(exp(x) + exp(-x)))
recurrent_activation = 'sigmoid' #sigmoid(x) = 1 / (1 + exp(-x))     #hard_signmoid?????
output_activation = 'linear'

batch_size = 32 #number of samples trained before performing one step of gradient update for the loss function (default is stochastic gradient descent)
look_back = 50 #how many frames will be used as a history for prediction
offset = 1 #how many frames in the future is the prediction going to occur at
forecast = 1 #how many frames will be predicted
sample_increment = 25 #number of frames between each sample
epochs = 30 #maximum number of times all training samples are fed into the model for training
units = 1000 #number of nodes in each hidden layer of the network
metrics=['accuracy']
stopping_patience = 5

temperature = 1
sigma_temp = 0.01
mixtures = 10


n_features = 165 #number of columns in the input dimension
frames = 500 #number of frames the model should generate
training_split = 0.7 #the proportion of the data to use for training
validation_split = 0.2 #the proportion of the data to use for validating during the training phase at the end of each epoch
evaluation_split = 0.1 #(test_split) the proportion of the data for evaluating the model effectiveness after training completion
convensional_method = False #whether or not the rotational dataset should be standardized by convensional methods (val-mean)/std vs. val/180

csv_data_dir = "/Akamai/MLDance/data/CSV/Raw" #directory to the csv representation of the dances
np_data_dir = "/Akamai/MLDance/data/Numpy" #directory to the numpy representation of the dances
logs_dir = "../logs" #general output directory

data_identifier = "lb-{}_o-{}_f-{}_si-{}_sm-{}".format(look_back, offset,forecast, sample_increment,convensional_method) #data-specific string for use in creating readily identifiable filenames
model_identifier = "units-{}_timesteps-{}".format(units, look_back) #model-specific string for use in creating readily identifiable filenames
split_identifier = "ts-{}_vs-{}_es-{}".format(training_split, validation_split, evaluation_split) #split-specific string for use in creating readily identifiable filenames
full_identifier = model_identifier+"_"+data_identifier+"_"+split_identifier

np_save_dir = os.path.join(np_data_dir, model_identifier) #output directory for model-specific numpy content
logs_save_dir = os.path.join(logs_dir, model_identifier) #output directory for model-specific non-numpy content

checkpoint_filename = "weights_{}_".format(look_back)+full_identifier+"_epoch-{epoch:02d}_loss-{loss:.2f}_acc-{accuracy:.2f}_val-loss-{val_loss:.2f}_val-acc-{val_accuracy:.2f}.h5"
checkpoint_filepath = os.path.join(logs_save_dir, checkpoint_filename)
architecture_filepath = os.path.join(logs_save_dir, "model-architecture_"+model_identifier+".h5")