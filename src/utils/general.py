from lib.general_dependencies import *
from lib import global_variables

def write(output="", file=sys.stdout):
    """ Display a string to an output stream. This allows for status updates to be outputted to a file. Default is a console log.

    :param output: total length of the progress bar
    :type str
    :param file: what to display/write the progress bar to
    :type output stream
    """
    file.write(output)
    file.write("\n")
    file.flush()
    
def json_to_file(filepath, json_object):
    with open(filepath, "w") as file:  
        json.dump(json_object, file) 
    
def progressbar(it, prefix="", size=60, file=sys.stdout):
    """ Create a visualization of a progress bar updates according to completion status

    :param it: job you are trying to create a progress bar for
    :type obj (sequence or collection)
    :param prefix: The text to display to the left of the status bar
    :type str
    :param size: total length of the progress bar
    :type int
    :param file: what to display/write the progress bar to
    :type output stream
    :return: job you are trying to create a progress bar for
    :rtype: obj (sequence or collection)
    """
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()
    
def create_dir(path):
    """ Create the cooresponding directory files for the given path if it does not yet exist

    :param path: proposed directory filepath
    :type str
    :return: created directory filepath
    :rtype: str
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def get_unique_dance_names(csv_path):
    """ Aggregate the names of unique dances
    
    :param csv_path: directory to the csv representation of the dances
    :type str
    :return: the dance names where there are csv files for
    :rtype: list
    """
    filenames = [f for f in os.listdir(csv_path) if f.endswith('.csv')]
    for file in enumerate(filenames): #enumerating creates an array where 0 corresponds to the index of the file in filenames and 1 corresponds to the filename
        filenames[file[0]] = '_'.join(file[1].split("_")[:-1])
    return list(set(filenames))

def get_save_path(filename):
    """ Identify the full path (directory, filename, extention) to save the csv and numpy data to

    :param filename: the name of the dance to grab the data from
    :type str
    :return: the csv and numpy save filepaths
    :rtype: tuple
    """
    csv_filename = os.path.join(global_variables.csv_data_dir, filename)
    np_filename = os.path.join(global_variables.np_save_dir, filename+"_"+global_variables.data_identifier+"_ts-{}".format(global_variables.training_split))
    return csv_filename, np_filename