from lib.generator_dependencies import *
from lib.general_dependencies import *
from .model_helper import *
from utils import *
from lib.global_variables import training_split, convensional_method, history_eval_file, logs_save_dir, look_back
from .custom_callback import *

    
def _save_generated_dance(generated_data, original_filename, save_filename):
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
    save_generated_dance(generated_data, training_split, hierarchy_file, c_headers, save_filename, convensional_method)  

def benchmark(model, eval_X, eval_Y, out_file=sys.stdout):
    """ Runs the evaluation data through the model to obtain the overall metrics on data the model did not train on

    :param model
    :type numpy.Array
    :param eval_X
    :type str
    :param save_filename: the directory and filename to store the generated dance at
    :type str
    :return: the training metric information (dict) and the model instance
    :type tuple
    """
    history = model.evaluate(eval_X, eval_Y, callbacks=[CustomCallback(out_file)], verbose=1, return_dict=True)
    with open(history_eval_file, "w") as history_file:  
        json.dump(history, history_file) 
    write("Saved evaluation metric history to json file:\n\t"+history_eval_file, out_file)
    print("Saved evaluation metric history to json file:\n\t"+history_eval_file)
    return history, model

def generate_dance(model, n_frames, random_frame=False, out_file=sys.stdout):
    """ Generate a dance sequence with the given model

    :param model: the model to use for prediction 
    :type keras.Model
    :param n_frames: the number of frames the model should generate
    :type int
    """
    #select random dance for seed
    dances = get_unique_dance_names(csv_data_dir)
    seed_dance_index = random.randint(0, len(dances) - 1)
    csv_filename, np_filename = get_save_path(dances[seed_dance_index])
    dance = get_processed_data(csv_filename, np_filename, training_split, convensional_method)
    
    seed = dance[:look_back]
    if random_frame:
        #select random frame(s) for seed
        seed_frame_index = random.randint(0, len(dance) - (look_back+1))
        seed = dance[seed_frame_index:seed_frame_index+look_back]
    
    write("Generating dance with seed from "+dances[seed_dance_index], out_file)
    write("Generating dance with seed from "+dances[seed_dance_index]) #sys.stdout
    #for diversity in [0.2, 0.5, 1.0, 1.2]:
    for diversity in [1.0]:
        start_time = time.time()
        write("{}-diversity Predicting Beginning:\t".format(diversity)+ time.ctime(start_time), out_file)
        write("{}-diversity Predicting Beginning:\t".format(diversity)+ time.ctime(start_time)) #sys.stdout
        generated = seed
        for i in progressbar(range(n_frames),"{} Progress: ".format(diversity)):
            preds = model.predict(np.array([generated[-look_back:]]), verbose=0)[0]
            #mdn.sample_from_output(preds, feature_size, mixtures, temp=diversity, sigma_temp=sigma_temp)
            generated = np.vstack((generated, preds))
        filename = os.path.join(logs_save_dir, "generated_dance_{}-frames_{}-diversity".format(n_frames, diversity))
        _save_generated_dance(generated, dances[seed_dance_index], filename)
        write("Generation Complete --- %s minutes ---" % ((time.time() - start_time)/60), out_file)
        write("Saved generated dance position and rotation csv:\n\t"+ filename, out_file)
        write("Generation Complete --- %s minutes ---" % ((time.time() - start_time)/60)) #sys.stdout
        write("Saved generated dance position and rotation csv:\n\t"+ filename) #sys.stdout
        