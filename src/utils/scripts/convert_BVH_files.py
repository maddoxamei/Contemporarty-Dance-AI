import os
from lib.global_variables import _extras_dir, csv_data_dir

source_dir = os.path.join(_extras_dir, r"data/BVH")

def _get_bvh_filenames():
    filenames = [f for f in os.listdir(source_dir) if f.endswith('.bvh')]
    for file in enumerate(filenames):
        filenames[file[0]] = file[1][:-4]
    return set(filenames)

def _bvh_to_csv(filename, save_dir):
    os.system('bvh-converter -r '+os.path.join(source_dir,filename)+".bvh") #os.system
    hierarchy_file = os.path.join(os.path.join(save_dir,"hierarchy"),filename.split('_')[0]+"_hierarchy.csv")
    if not os.path.isfile(hierarchy_file):
        os.system('bvh2csv -H '+os.path.join(source_dir,filename)+".bvh")
        os.system('mv '+os.path.join(source_dir,filename+"_hierarchy.csv ")+hierarchy_file)
        os.system('rm '+os.path.join(source_dir,filename+"_pos.csv "))
        os.system('rm '+os.path.join(source_dir,filename+"_rot.csv "))

def _convert_BVH_files(save_dir):
    for file in _get_bvh_filenames():
        _bvh_to_csv(file, save_dir)
    os.system('mv *.csv '+save_dir)
    
def csv_to_bvh(hierarchy_file, position_file, rotation_file):
    os.system('csv2bvh '+hierarchy_file+' '+position_file+' '+rotation_file)

if __name__ == "__main__":
    _convert_BVH_files(csv_data_dir)