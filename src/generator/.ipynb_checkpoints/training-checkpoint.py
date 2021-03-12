from lib.general_dependencies import *
from lib.data_dependencies import *
from lib.generator_dependencies import *
from .model_helper import *
from utils import get_unique_dance_names, get_sample_data, write, progressbar, get_save_path
from lib.global_variables import *
from .custom_callback import *

        
def train_model(model, out_file=sys.stdout):
    """ Trains the model with the dance data.
        The History object's History.history attribute is a record of training loss values and metrics values at successive epochs, 
            as well as cooresponding validation values (if applicable).  

    :param model: the model to train
    :type keras.Model
    :param out_file: what to display/write the status information to
    :type output stream
    :return: the class containing the training metric information, the trained model, and the comprehensive evaluation data
    :type tuple
    """
    
    dances = get_unique_dance_names(csv_data_dir)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=weights_file, monitor='val_loss', mode='auto', save_weights_only=True, save_best_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=stopping_patience, verbose=2, mode='auto', restore_best_weights=True)
    callbacks_list = [keras.callbacks.TerminateOnNaN(), checkpoint, early_stopping, CustomCallback(out_file)]

    comprehensive_train_X = np.array([])
    comprehensive_train_Y = np.array([])
    comprehensive_validate_X = np.array([])
    comprehensive_validate_Y = np.array([])
    comprehensive_evaluation_X = np.array([])
    comprehensive_evaluation_Y = np.array([])
    
    write("Fetching and Agregating Training Data ...") #sys.stdout
    start_time = time.time()
    for dance in progressbar(dances, "Progress: "):
        csv_filename, np_filename = get_save_path(dance)
        train_X, train_Y, validate_X, validate_Y, evaluation_X, evaluation_Y = get_sample_data(csv_filename, np_filename, look_back, offset, forecast, sample_increment, training_split, validation_split, convensional_method)
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
    write("Fetching and Agregating Training Data --- {} seconds ---".format(start_time - time.time()), out_file)  
    start_time = time.time()
    history = model.fit(comprehensive_train_X, comprehensive_train_Y, 
                        batch_size = batch_size, 
                        callbacks=callbacks_list, 
                        validation_data= (comprehensive_validate_X, comprehensive_validate_Y),
                        epochs=epochs, 
                        shuffle = shuffle_data,
                        verbose=1)
    
    save_model_checkpoint(model, model_file)
    np.save(evaluation_filepath+"_X", comprehensive_evaluation_X)
    np.save(evaluation_filepath+"_Y", comprehensive_evaluation_Y)
    with open(history_train_file, "w") as history_file:  
        json.dump(pd.DataFrame.from_dict(history.history).to_dict(), history_file) 
    write("Saved training metric history to json file:\n\t"+history_train_file) #sys.stdout
    write("Saved training metric history to json file:\n\t"+history_train_file, out_file)
    return history, model, comprehensive_evaluation_X, comprehensive_evaluation_Y