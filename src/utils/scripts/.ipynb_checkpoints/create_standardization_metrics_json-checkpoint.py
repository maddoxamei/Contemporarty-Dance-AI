from lib import * #import dependancies and global_variables
import utils
from utils.data import _pre_process_pos_data

def get_comprehensive_train_data(train_split, vert_axis):
    dances = utils.get_unique_dance_names(csv_data_dir)
    comprehensive_train = np.array([])
    for dance in utils.progressbar(dances,"{}-{}".format(train_split, vert_axis)):
        data = get_data_for_standardization(dance, vert_axis)
        train_data = data.copy()[0:int(len(data)*train_split)]
        if(len(comprehensive_train)==0):
            comprehensive_train = train_data
        else:
            comprehensive_train = np.vstack((comprehensive_train,train_data))
    return comprehensive_train

def get_data_for_standardization(filename, vert_axis):
    csv_filename, np_filename = utils.get_save_path(filename)
    position_df = pd.read_csv(csv_filename+"_worldpos.csv", usecols=['Hips.X','Hips.Y','Hips.Z'])
    rotation_df = pd.read_csv(csv_filename+"_rotations.csv")
    
    position_df = _pre_process_pos_data(position_df, vert_axis)
    
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

    for split in [i/10 for i in range(1,11)]:
        vert_dict = {}
        train_X = pd.DataFrame(get_comprehensive_train_data(split, 'X'), columns=c_headers)
        train_Y = pd.DataFrame(get_comprehensive_train_data(split, 'Y'), columns=c_headers)
        train_Z = pd.DataFrame(get_comprehensive_train_data(split, 'Z'), columns=c_headers)
        train_None = pd.DataFrame(get_comprehensive_train_data(split, None), columns=c_headers)

        vert_dict.update({"X":{"mean":train_X.mean().to_dict(), "std":train_X.std().to_dict()}})
        vert_dict.update({"Y":{"mean":train_Y.mean().to_dict(), "std":train_Y.std().to_dict()}})
        vert_dict.update({"Z":{"mean":train_Z.mean().to_dict(), "std":train_Z.std().to_dict()}})
        vert_dict.update({"None":{"mean":train_None.mean().to_dict(), "std":train_None.std().to_dict()}})

        _dict.update({split:vert_dict})

    with open("standardization_metrics.json", 'w') as f:
        json.dump(_dict, f)
        
if __name__ == "__main__":
    create_standardization_json()