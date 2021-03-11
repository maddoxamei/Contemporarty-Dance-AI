import pandas as pd
import os, sys

source_dir = "../Data/CSV/Raw/"
save_dir = "../Data/CSV/Pre-Processed/"

def getFileNames():
        filenames = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
        for file in enumerate(filenames):
        	name = file[1].split('_')
        	name.pop(len(name)-1)
        	filenames[file[0]] = '_'.join(name)
        return set(filenames)

def process_data(filename):
	print(filename)
	pos_data = pd.read_csv(source_dir+filename+"_worldpos.csv")
	rot_data = pd.read_csv(source_dir+filename+"_rotations.csv")

	'''
	col_list = pos_data.columns
	col_list = set(col_list)
	end = [s for s in list_2 if "end" in s]
	for col in end:
		pos_data.pop(col)
		rot_data.pop(col)
	'''


	#normalization force values from -1 to 1
	rot_data = rot_data/180.0

	#Add the root (hip) data for spacial movement
	rot_data['Hips.Pos.X'] = pos_data.pop('Hips.X')
	rot_data['Hips.Pos.Y'] = pos_data.pop('Hips.Y')
	rot_data['Hips.Pos.Z'] = pos_data.pop('Hips.Z')

	#Making movement relative to an origin of 0,0,0 for consistancy within different dances
	rot_data['Hips.Pos.X'] = rot_data['Hips.Pos.X'] + (-1*rot_data['Hips.Pos.X'][0])
	rot_data['Hips.Pos.Y'] = rot_data['Hips.Pos.Y'] + (-1*rot_data['Hips.Pos.Y'][0])
	rot_data['Hips.Pos.Z'] = rot_data['Hips.Pos.Z'] + (-1*rot_data['Hips.Pos.Z'][0])


	raw = rot_data.copy()
	relative = rot_data.copy()

	previous = next(relative.iterrows())[1]
	for index, row in relative.iterrows():
		if(index != 0):
			current = raw.iloc[index]
			relative.iloc[index] = row - previous
			previous = current
	relative = relative.drop([0])

	#time = rot_data.pop('Time') #maybe change to time change value instead? To indicate speed)
	raw['Sentiment'] = pd.to_numeric(filename[-1])
	relative['Sentiment'] = pd.to_numeric(filename[-1])

	combined = pd.concat([raw.drop([0]), relative], axis=1, sort=False)

	#raw.to_csv(save_dir+"Absolute/"+filename+".csv", index=False)
	combined.to_csv(save_dir+"Combined/"+filename+".csv", index=False)
	relative.to_csv(save_dir+"Relative/"+filename+".csv", index=False)


def create_comprehensive(location):
	files = list(getFileNames())

	comprehensive = pd.read_csv(save_dir+location+files[0]+".csv")
	addition = []
	for file in files:
		if(file != files[0]):
			print(file)
			addition = pd.read_csv(save_dir+location+file+".csv")
			comprehensive = comprehensive.append(addition, ignore_index=True, sort=False)
	comprehensive.to_csv(save_dir+location+"_comprehensive_"+".csv", index=False)

'''
for file in getFileNames():
	process_data(file)

print("Absolute")
create_comprehensive("Absolute/")
print("Combined")
create_comprehensive("Combined/")
print("Relative")
create_comprehensive("Relative/")
'''
