from lib.generator_dependencies import *
from lib.general_dependencies import *

def model_summary(model):
    """ Convert the model architecture summary to a writeable format

    :param model: the model to display the summary of
    :type keras.Model
    :return: the visual diagram of the model
    :type list
    """
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    return "\n".join(stringlist)
    
def compile_model(model, optimizer, loss, metrics):
    """ Compile the given model so that it is ready for training and/or prediction/evaluation

    :param model: the model to compile
    :type keras.Model
    :return: the compiled model
    :type keras.Model
    """
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)#tf.keras.metrics.MeanSquaredError(),
    return model

def save_architecture(model, out_path):
    """ Save the architecture (layers and how they are connected*). 
        Model can be created with a freshly initialized state for the weights and no compilation information from this savefile

    :param model: the model to save
    :type keras.Model
    :param out_path: full path (directory+filename+file-extension) to save the model architecture to
    :type str
    """
    json_config = model.to_json()
    with open(out_path, "w") as file:  
        json.dump(json_config, file) 
    
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
    
def save_model_checkpoint(model, file):
    """ Save the entire model. Model can be loaded and restart training right where you left off
        The following are saved:
            weight values
            Model's architecture
            Model's training configuration (what you pass to the .compile() method)
            Optimizer and its state, if any (this allows you to restart training)

    :param model: the model to save
    :type keras.Model
    :param file: the path+filename+extension to save the model to
    :type str
    """
    model.save(file)
    
def load_architecture(file):
    """ Load the architecture (layers and how they are connected*). 
        Model can be created with a freshly initialized state for the weights.
        There is NO compilation information in this savefile.

    :param file: .json file which holds the model's architecture data
    :type str
    :return: the model's architecture
    :type keras.Model
    """
    with open(file, 'r') as json_file:
        json_info = json_file.read()
    return keras.models.model_from_json(json.loads(json_info))

def load_weights(model, file):
    model.load_weights(file)
    return model

def load_trained_model(architecture_file, weights_dir):
    """ Load the "best" model for training/prediction/evaluation. Optimizer state is NOT included so training will be "restarted"

    :param file: the model's architecture
    :type str
    :param file: the directory storing the model's weight checkpoints
    :type str
    :return: the compiled model
    :type keras.Model
    """
    model = load_architecture(architecture_file)
    weight_files = [f for f in os.listdir(weights_dir) if 'weight' in f]
    weight_files.sort()
    return load_weights(model, os.path.join(weights_dir, weight_files[-1]))

def load_model_checkpoint(file):
    """ Load the entirety of a model (architecture, weights, training config, and optimizer state). Compiled when loaded so training/prediction/evaluation can be restarted right where the model left off. 

    :param file: .h5 file which holds the model's information
    :type str
    :return: the compiled model
    :type keras.Model
    """
    return keras.models.load_model(file, compile=True)
