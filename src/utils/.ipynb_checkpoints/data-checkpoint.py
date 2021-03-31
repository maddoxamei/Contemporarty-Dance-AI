from lib.data_dependencies import *
from lib.general_dependencies import *
from lib.global_variables import standardization_json, vertical_spacial_axis, standardize_positions, relativize_positions

def _drop_excess_features(df, axis, dropped_features = ['End', 'Time']):
    df_features = df.index if axis==0 else df.columns
    for feature in [f for f in df_features for remove in dropped_features if remove in f]:
        df.drop(feature, axis=axis, inplace=True)
    return df
        
def _standardize_dataset(data_df, training_split, vertical_axis = None):
    with open(standardization_json, 'r') as f:
        metrics_dict = f.read()
    metrics_dict = json.loads(metrics_dict)[str(training_split)][str(vertical_axis)]
    full_metrics_df = pd.DataFrame.from_dict(metrics_dict)
    # yields the elements in `list_2` that are NOT in `list_1`
    metrics_df = _drop_excess_features(full_metrics_df.copy(), 0, np.setdiff1d(full_metrics_df.index, data_df.columns))
    return (data_df - metrics_df['mean']) / metrics_df['std']

def _un_standardize_dataset(data_df, training_split, vertical_axis = None):
    with open(standardization_json, 'r') as f:
        metrics_dict = f.read()
    metrics_dict = json.loads(metrics_dict)[str(training_split)][str(vertical_axis)]
    full_metrics_df = pd.DataFrame.from_dict(metrics_dict)
    # yields the elements in `list_2` that are NOT in `list_1`
    metrics_df = _drop_excess_features(full_metrics_df.copy(), 0, np.setdiff1d(full_metrics_df.index, data_df.columns))
    return (data_df * metrics_df['std']) + metrics_df['mean']

def _relativize_data(data):
    """ Linearly transform the data such that each frame represents the "change" from the previous frame.   
    
    :param data: the data to linearly transform
    :type pandas.DataFrame
    :return: the relative representation of the data
    :rtype: pandas.DataFrame
    """
    next_frame = data.iloc[len(data)-2] #second to last frame
    for index, row in data.iloc[::-1].iterrows(): #itterating through reverse frame order
        if(index == 0):
            data.iloc[index] = 0
        else:
            data.iloc[index] = row - data.iloc[index-1]
    return data

def _un_relativize_data(data):
    """ Linearly transform relative data such that each frame represents an absolute/global value
    
    :param data: relative representation of the dance frames
    :type pandas.DataFrame
    :return: absolute/global representation of the dance frames
    :rtype: pandas.DataFrame
    """
    previous_frame = 0
    for index, row in data.iterrows(): #itterating through frame order
        if(index > 0):
            data.iloc[index] = row + data.iloc[index-1]
    return data

def _pre_process_pos_data(position_df, relativize, standardize, training_split, vertical_axis = None):
    """ Alter the position data such that the horizontal planar movement is relative and the verticle axis remains global
    
    :param position_df: Hips.X, Hips.Y, and Hips.Z position data
    :type pandas.DataFrame
    :param training_split: the proportion of the data to use for training
    :type float
    :param standardize: whether or not the position data should be standardized
    :type bool
    :param vertical_axis: the header cooresponding to the verticle axis (either X, Y, Z) or None
    :type char
    :return: dataframe that contains the altered hip positions
    :rtype: pandas.DataFrame
    """
    c_headers = ['.'.join(c.split('.')[0:1]+['Pos']+c.split('.')[1:2]) if 'Pos' not in c else c for c in position_df.columns]
    prefix = '.'.join(position_df.columns[0].split('.')[:-1]+[""])
    if(relativize):
        if(vertical_axis):
            vertical_movement = position_df.pop(prefix+vertical_axis)
            position_df = _relativize_data(position_df)
            position_df[prefix+vertical_axis] = vertical_movement
        else:
            position_df = _relativize_data(position_df)
    if(standardize):
        position_df = _standardize_dataset(position_df, training_split, vertical_axis)
    return position_df

def _post_process_pos_data(position_df, hierarchy_df, relativized, standardized, training_split, vertical_axis = None):
    """ Alter the position data such that the horizontal planar movement is global, making all position channel dimensions global
    
    :param position_df: Hips.X, Hips.Y, and Hips.Z position data
    :type pandas.DataFrame
    :param hierarchy: joint offset data (must include Hips.X, Hips.Y, and Hips.Z )
    :type pandas.DataFrame
    :param standardized: whether or not the position data should be standardized
    :type bool
    :param vertical_axis: the header cooresponding to the verticle axis (either X, Y, Z) or None
    :type char
    :param training_split: the proportion of the data to use for training
    :type float: the header cooresponding to the verticle axis (either X, Y, Z)
    :type char
    :return: dataframe that contains the altered hip positions
    :rtype: pandas.DataFrame
    """
    c_headers = ['.'.join(c.split('.')[0:1]+['Pos']+c.split('.')[1:2]) if 'Pos' not in c else c for c in position_df.columns]
    prefix = '.'.join(position_df.columns[0].split('.')[:-1]+[""])
    if(standardized):
        position_df = _un_standardize_dataset(position_df, training_split, vertical_axis)
    if(relativized):
        if(vertical_axis):
            vertical_movement = position_df.pop(prefix+vertical_axis)
            position_df = _un_relativize_data(position_df)
            position_df[prefix+vertical_axis] = vertical_movement
        else:
            position_df = _un_relativize_data(position_df)
    
    return position_df

def _pre_process_rot_data(rotation_df, standard_method, training_split, vertical_axis = None):
    """ Standardizes the rotational dataset values.
        Standard method by mean subtraction and division by the standard deviation along each dimension. This will center the values around zero.
        Unconvensional method by dividing by 180 as all rotational values are constrained between -180 and 180
    
    :param rotation_df: the data containing the rotational channels of the dance frames
    :type pandas.DataFrame
    :param standard_method: whether or not the rotational dataset should be standardized by convensional methods
    :type bool
    :param training_split: the proportion of the data to use for training
    :type float
    :param vertical_axis: the header cooresponding to the verticle axis (either X, Y, Z) or None
    :type char
    :return: the standardized rotational dance frames
    :rtype: pandas.DataFrame
    """
    if standard_method:
        rotation_df = _standardize_dataset(rotation_df, training_split, vertical_axis)
    else:
        rotation_df = rotation_df/180
    return rotation_df

def _post_process_rot_data(rotation_df, standard_method, training_split, vertical_axis = None):
    """ Undoes the transformations during the rotational data standardization process.
        Values will no longer be centered around 0.
    
    :param rotation_df: the data containing the rotational channels of the dance frames
    :type pandas.DataFrame
    :param training_split: the proportion of the data to use for training
    :type float
    :param standard_method: whether or not the rotational dataset should be standardized by convensional methods
    :type bool
    :param vertical_axis: the header cooresponding to the verticle axis (either X, Y, Z) or None
    :type char
    :return: the un-standardized rotational dance frames
    :rtype: pandas.DataFrame
    """
    if standard_method:
        rotation_df = _un_standardize_dataset(rotation_df, training_split, vertical_axis)
    else:
        rotation_df *= 180
    return rotation_df

def _pre_process_data(csv_filename, training_split, standard_method):
    """ Process the data so that it is ready to be fed into the neural network
    
    :param csv_filename: path (directory+filename) to the csv representation of a particular dance (does NOT includes file-extension)
    :type str
    :param training_split: the proportion of the data to use for training
    :type float
    :param standard_method: whether or not the rotational dataset should be standardized by convensional methods
    :type bool
    :return: dataframe that contains the processed dance frames
    :rtype: pandas.DataFrame
    """   
    position_df = pd.read_csv(csv_filename+"_worldpos.csv", usecols=['Hips.X','Hips.Y','Hips.Z'])
    rotation_df = pd.read_csv(csv_filename+"_rotations.csv")
    
    data = _pre_process_rot_data(rotation_df.copy(), standard_method, training_split, vertical_spacial_axis)
    # Relativize the horizontal planer position movement
    position_df = _pre_process_pos_data(position_df, relativize_positions, standardize_positions, training_split, vertical_spacial_axis)
    
    # Remove the all the features which aren't all zeros and the time feature from the dataset
    _drop_excess_features(data, 1)
    
    # Add the root (hip) data for spacial movement
    data['Hips.Pos.X'] = position_df.pop('Hips.X')
    data['Hips.Pos.Y'] = position_df.pop('Hips.Y')
    data['Hips.Pos.Z'] = position_df.pop('Hips.Z')
    return data

def _post_process_data(rotation_df, position_df, hierarchy_df, training_split, standard_method):
    """ Un-process the data to transform the values representign the generated dance into something MotionBuidler (visualization program) can interpret.
        Un-standardize and un-realativaize the generated dance

    :param rotation_df: the AI generated rotational dance data (is processed) 
    :type pandas.DataFrame
    :param position_df: the AI generated positional dance data (is processed) 
    :type pandas.DataFrame
    :param hierarchy: joint offset data (must include Hips.X, Hips.Y, and Hips.Z )
    :type pandas.DataFrame
    :param training_split: the proportion of the data to use for training
    :type float
    :param standard_method: whether or not the rotational dataset should be standardized by convensional methods
    :type bool
    :return: the unprocessed versions of the rotation and position frames from the generated dance
    :rtype: tuple
    """
    #undo the normalization and standardization of the data
    rotation_df = _post_process_rot_data(rotation_df, standard_method, training_split, vertical_spacial_axis)
    position_df = _post_process_pos_data(position_df, hierarchy_df, relativize_positions, standardize_positions, training_split, vertical_spacial_axis)
    
    new_headers = []
    joints = [j for j in hierarchy_df['joint'].to_numpy() if "End" not in j]
    for j in joints:
        new_headers.append(j+".Z")
        new_headers.append(j+".X")
        new_headers.append(j+".Y")

    rotation_df = rotation_df.reindex(columns=new_headers)  
    
    rotation_df.insert(0, 'time', np.arange(0.0, len(rotation_df))*0.03333333)
    position_df.insert(0, 'time', np.arange(0.0, len(position_df))*0.03333333)
    
    return rotation_df, position_df

def get_processed_data(csv_filename, np_filename, training_split, standard_method):
    """ Fetch the pre-procced data cooresponding to the given dance name

    :param csv_filename: path (directory+filename) to the csv representation of a particular dance (does NOT includes file-extension)
    :type str
    :param np_filename: path (directory+filename) to the numpy representation of a particular dance (does NOT includes file-extension)
    :type str
    :param training_split: the proportion of the data to use for training
    :type int
    :param standard_method: whether or not the rotational dataset should be standardized by convensional methods
    :type bool
    :return: the pre-processed dance data
    :rtype: numpy.ndarray
    """
    #If the corresponding numpy file doesn't yet exist, create and save it
    if not (os.path.exists(np_filename+".npy")):
        #Print statement for status update
        #print("Creating pre-processed datafile:", np_filename)
        #load the csv file and establish the number of rows and columns
        data = _pre_process_data(csv_filename, training_split, standard_method)
        
        #data.to_csv("preprocessed_data.csv", index=False)
        #data = data.iloc[:].values #Enables selection/edit of cells in the pandas dataframe
        np.save(np_filename, data)
        #print("Saved the pre-processed data to\n\t", np_filename)

    return np.load(np_filename+".npy")

def save_generated_dance(generated_data, training_split, hierarchy_file, c_headers, save_filename, standard_method):
    """ Save the generated dance to a csv file for bvh converstion later.

    :param generated_data: the array corresponding the the generated frames of dance
    :type numpy.ndarray
    :param training_split: the proportion of the data to use for training
    :type int
    :param hierarcy_file: full path (directory+filename+file-extension) to csv hierarchy representation of the dance joints
    :type str
    :param c_headers: list of original column headers from the dance the seed is from, minus the time
    :type list
    :param save_filename: the directory and filename to store the generated dance at
    :type str:
    :param standard_method: whether or not the rotational dataset should be standardized by convensional methods
    :type bool
    """
    rotation = generated_data[:,:162] #get the first 162 columns
    position = generated_data[:,-3:] #get the last 3 columns
    hierarchy = pd.read_csv(hierarchy_file)
    
    rotation_df = pd.DataFrame(rotation, columns=c_headers)
    position_df = pd.DataFrame(position, columns=c_headers[:3])
    
    rotation_df, position_df = _post_process_data(rotation_df, position_df, hierarchy, training_split, standard_method)
 
    rotation_df.to_csv(save_filename+"_rot.csv", index=False)
    position_df.to_csv(save_filename+"_pos.csv", index=False) 
    
    return position_df, rotation_df
    

def _split_data(data, training_split, validation_split):
    """ Separates the data into three different datasets (training, validation, and evaluation) based off the pre-defined split proportions.
        Each consecutive dataset is selected from the last samples of the previous one.
        
        Data is NOT randomly shuffled before spliting to ensure...
            chopping the data into windows of consecutive samples is still possible
            validation/test results are more realistic, being evaluated on data collected after the model was trained

    :param data: the processed rot or pos data for a particular dance, not yet sequenced/turned into samples
    :type numpy.ndarray
    :param training_split: the proportion of the data to use for training
    :type float
    :param validation_split: the proportion of the data to use for validating during the training phase at the end of each epoch
    :type float
    :return: the spliced datasets cooresponding to training, validation, and evaluation, respectively
    :rtype: tuple
    """
    train_index = int(len(data)*training_split)
    validation_index = int(len(data)*(training_split+validation_split))
    train_data = data.copy()[0:train_index]
    validate_data = data.copy()[train_index:validation_index]
    evaluate_data = pd.DataFrame()
    if(validation_index<len(data)):
        evaluate_data = data.copy()[validation_index:]
    return train_data, validate_data, evaluate_data
 
def _sequence_data(data, look_back, offset, forecast, sample_increment):
    """ Create samples (input X and target Y) from the given data

    :param data: the data to create samples from
    :type numpy.ndarray
    :param look_back: how many frames will be used as a history for prediction
    :type int
    :param offset: how many frames in the future is the prediction going to occur at
    :type int
    :param forecast: how many frames will be predicted
    :type int
    :param sample_increment: number of frames between each sample
    :type int
    :return: the collection of input X and target Y samples as numpy arrays
    :type tuple
    """
    #load the csv file and establish the number of rows and columns
    N_ROWS = data.shape[0]
    N_COLOMNS = data.shape[1]

    #data = data.iloc[:].values #Enables selection/edit of cells in the dataset
    dataX = []
    dataY = []
        
    #Generate the sequences
    for i in range(0, N_ROWS - look_back - offset - forecast + 2, sample_increment): #range(start, stop, step) 
        # Create an input sample cooresponding to the sequence of {look_baack} frame(s) starting at {i}
        seqIn = data[i: i+look_back] 
        # Create an output sample cooresponding to the sequence of {forcast} frame(s) starting at {offset} frame(s) in the future
        seqOut = data[i+look_back+offset-1 : i+look_back+offset+forecast-1] 
        dataX.append(seqIn)
        dataY.append(seqOut)

    #X shape [samples, timesteps, features]
    #Y shape [samples, 1, features]
    X, Y = np.array(dataX), np.array(dataY)

    N_SAMPLES = len(dataX)
    Y = np.reshape(Y, (N_SAMPLES, N_COLOMNS))
    return X, Y
    
def get_sample_data(csv_filename, np_filename, look_back, offset, forecast, sample_increment, training_split, validation_split, standard_method):
    """ Fetch the pre-sequenced or sampled data from the given dance, or create/save it if it does not yet exist

    :param csv_filename: path (directory+filename) to the csv representation of a particular dance (does NOT includes file-extension)
    :type str
    :param np_filename: path (directory+filename) to the numpy representation of a particular dance (does NOT includes file-extension)
    :type str
    :param look_back: how many frames will be used as a history for prediction
    :type int
    :param offset: how many frames in the future is the prediction going to occur at
    :type int
    :param forecast: how many frames will be predicted
    :type int
    :param sample_increment: number of frames between each sample
    :type int
    :param training_split: the proportion of the data to use for training
    :type float
    :param validation_split: the proportion of the data to use for validating during the training phase at the end of each epoch
    :type float
    :param standard_method: whether or not the rotational dataset should be standardized by convensional methods
    :type bool
    :return: the collection of input X and target Y samples for the train, validation, and evaluation datasets
    :type tuple
    """    
    #Establish filenames (X is for input, Y is for expected output)
    training_save_X = np_filename+"_train-X"
    training_save_Y = np_filename+"_train-Y"
    
    validation_save_X = np_filename+"_val-X"
    validation_save_Y = np_filename+"_val-Y"
    
    evaluation_save_X = np_filename+"_test-X"
    evaluation_save_Y = np_filename+"_test-Y"
    
    # If the corresponding numpy file doesn't yet exist, create and save it
    if not (os.path.exists(training_save_X+".npy") and 
            os.path.exists(training_save_Y+".npy") and 
            os.path.exists(validation_save_X+".npy") and 
            os.path.exists(validation_save_Y+".npy") and 
            os.path.exists(evaluation_save_X+".npy") and 
            os.path.exists(evaluation_save_Y+".npy")):
        # Print statement for status update
        #print("Creating the sequenced data:", np_filename)
        
        # Preprocess the data, then split it into train, validation, and evaluation datasets
        data = get_processed_data(csv_filename, np_filename, training_split, standard_method)
        train_data, validate_data, evaluate_data = _split_data(data, training_split, validation_split)
        
        #np_to_csv(train_data, "train_data")
        #np_to_csv(evaluate_data, "evaluate_data")

        # Sequence the datasets and turn it into samples of X and Y for each
        train_X, train_Y = _sequence_data(train_data, look_back, offset, forecast, sample_increment)
        validate_X, validate_Y = _sequence_data(validate_data, look_back, offset, forecast, sample_increment)
        evaluation_X, evaluation_Y = _sequence_data(evaluate_data, look_back, offset, forecast, sample_increment)
        
        #np_to_csv(train_Y, "train_data")
        #np_to_csv(evaluation_Y, "evaluate_data")
             
        # Save the sample data
        np.save(training_save_X, train_X)
        np.save(training_save_Y, train_Y)
        np.save(validation_save_X, validate_X)
        np.save(validation_save_Y, validate_Y)
        np.save(evaluation_save_X, evaluation_X)
        np.save(evaluation_save_Y, evaluation_Y)
        #print("Saved the sequenced data to\n\t", np_filename)

    return np.load(training_save_X+".npy"), np.load(training_save_Y+".npy"), np.load(validation_save_X+".npy"), np.load(validation_save_Y+".npy"), np.load(evaluation_save_X+".npy"), np.load(evaluation_save_Y+".npy")

def np_to_csv(arr, savefile):
    df = pd.DataFrame(arr) 
    df.to_csv(savefile+".csv", index=False)