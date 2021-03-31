from lib import * #import dependancies and global_variables
import utils
from utils.data import _pre_process_pos_data, _post_process_pos_data
from utils.compare_dataframes import compute_differences, print_header

"""
Standardized Only produces 0 error between the raw input data and the pre-then-post processed data.
Relative Only and Relative+Standardized produces the same error and the same post output (as it should since standardization should be self-cancelled)
"""

def compare_processed_error(vert_axis = None):
    dance_index = 92
    dances = utils.get_unique_dance_names(csv_data_dir)
    dances.sort()
    csv_filename, np_filename = utils.get_save_path(dances[dance_index])
    print(csv_filename)
    hierarchy_file = os.path.join(hierarchy_dir, "AI_hierarchy.csv")
    hierarchy_df = pd.read_csv(hierarchy_file)
    raw_position, raw_rotation = get_raw_data(dances[dance_index])
    c_headers = [c for c in raw_rotation.columns if 'End' not in c and 'Time' not in c]
    full_headers = [c for c in raw_rotation.columns if 'End' not in c and 'Time' not in c]
    full_headers.append('Hips.Pos.X')
    full_headers.append('Hips.Pos.Y')
    full_headers.append('Hips.Pos.Z')
    raw_position.columns = full_headers[-3:]
    
    print_header("Vertical Axis: {}".format(vert_axis))
    
    rel = _pre_process_pos_data(raw_position.copy(), True, False, training_split, vert_axis)
    position_df = _post_process_pos_data(rel, hierarchy_df, True, False, training_split, vert_axis)
    print(position_df.head())
    compute_differences(position_df, raw_position, "Relativized")

    rel = _pre_process_pos_data(raw_position.copy(), False, True, training_split, vert_axis)
    position_df = _post_process_pos_data(rel, hierarchy_df, False, True, training_split, vert_axis)
    print(position_df.head())
    compute_differences(position_df, raw_position, "Standardized")

    rel = _pre_process_pos_data(raw_position.copy(), True, True, training_split, vert_axis)
    position_df = _post_process_pos_data(rel, hierarchy_df, True, True, training_split, vert_axis)
    print(position_df.head())
    compute_differences(position_df, raw_position, "Relativized + Standardized")
    
if __name__ == "__main__":
    for axis in ['Y', None]:#
        compare_processed_error(axis)