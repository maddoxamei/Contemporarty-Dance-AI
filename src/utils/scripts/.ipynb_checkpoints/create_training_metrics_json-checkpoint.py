from lib import * #import dependancies and global_variables
import utils
from utils.data import _pre_process_pos_data, _pre_process_rot_data

def get_comprehensive_train_data(train_split, process):
    dances = utils.get_unique_dance_names(csv_data_dir)
    comprehensive_train = np.array([])
    for dance in utils.progressbar(dances,"{}-{}".format(train_split, process)):
        data = get_data(dance, process, train_split)
        train_data = data.copy()[0:int(len(data)*train_split)]
        if(len(comprehensive_train)==0):
            comprehensive_train = train_data
        else:
            comprehensive_train = np.vstack((comprehensive_train,train_data))
    return comprehensive_train

def get_raw_data(filename):
    csv_filename, np_filename = utils.get_save_path(filename)
    position_df = pd.read_csv(csv_filename+"_worldpos.csv", usecols=['Hips.X','Hips.Y','Hips.Z'])
    rotation_df = pd.read_csv(csv_filename+"_rotations.csv")
    
    #data = utils.get_processed_data(csv_filename, np_filename, training_split, processes, processes)
    data = rotation_df.copy()
    # Add the root (hip) data for spacial movement
    data['Hips.Pos.X'] = position_df.copy().pop('Hips.X')
    data['Hips.Pos.Y'] = position_df.copy().pop('Hips.Y')
    data['Hips.Pos.Z'] = position_df.copy().pop('Hips.Z')
    return data

def get_data(filename, process, train_split):
    csv_filename, np_filename = utils.get_save_path(filename)
    position_df = pd.read_csv(csv_filename+"_worldpos.csv", usecols=['Hips.X','Hips.Y','Hips.Z'])
    rotation_df = pd.read_csv(csv_filename+"_rotations.csv")
    
    #print(position_df.head())
    position_df = _pre_process_pos_data(position_df, process, train_split)    
    rotation_df = _pre_process_rot_data(rotation_df, process, train_split)
    #print(position_df.head())
    
    data = rotation_df.copy()
    # Add the root (hip) data for spacial movement
    data['Hips.Pos.X'] = position_df.copy().pop('Hips.X')
    data['Hips.Pos.Y'] = position_df.copy().pop('Hips.Y')
    data['Hips.Pos.Z'] = position_df.copy().pop('Hips.Z')
    return data

def create_standardization_json():
    _dict = {}

    dances = utils.get_unique_dance_names(csv_data_dir)
    original_data = pd.read_csv(os.path.join(csv_data_dir, dances[0]+"_rotations.csv"), nrows=0)
    c_headers = [c for c in original_data.columns]
    c_headers.append('Hips.Pos.X')
    c_headers.append('Hips.Pos.Y')
    c_headers.append('Hips.Pos.Z')
    
    for split in [i/10 for i in range(1,2)]:# 2-> 11
        vert_dict = {}
        
        train_X = pd.DataFrame(get_comprehensive_train_data(split, ""), columns=c_headers)
        vert_dict.update({" ":{"max":train_X.max().fillna(0).to_dict(), "mean":train_X.mean().fillna(0).to_dict(), "min":train_X.min().fillna(0).to_dict(), "std":train_X.std().fillna(0).to_dict()}})
        _dict.update({split:vert_dict})
        
        for p in "or":#[''.join(tups) for tups in itertools.permutations("osn", 3)]
            train_X = pd.DataFrame(get_comprehensive_train_data(split, p), columns=c_headers)
            vert_dict.update({p:{"max":train_X.max().fillna(0).to_dict(), "mean":train_X.mean().fillna(0).to_dict(), "min":train_X.min().fillna(0).to_dict(), "std":train_X.std().fillna(0).to_dict()}})

            _dict.update({split:vert_dict})

    with open(processing_json, 'w') as f:
        json.dump(_dict, f)

    for split in [i/10 for i in range(1,2)]:# 2-> 11
        vert_dict = {}
        
        for p in "sn":#[''.join(tups) for tups in itertools.permutations("osn", 3)]
            train_X = pd.DataFrame(get_comprehensive_train_data(split, p), columns=c_headers)
            vert_dict.update({p:{"max":train_X.max().fillna(0).to_dict(), "mean":train_X.mean().fillna(0).to_dict(), "min":train_X.min().fillna(0).to_dict(), "std":train_X.std().fillna(0).to_dict()}})

            _dict[split].update(vert_dict)

    with open(processing_json, 'w') as f:
        json.dump(_dict, f)
        
if __name__ == "__main__":
    create_standardization_json()