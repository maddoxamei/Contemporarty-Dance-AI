import os

source_dir = "../Data/BVH/"

def getBvhNames():
        filenames = [f for f in os.listdir(source_dir) if f.endswith('.bvh')]
        for file in enumerate(filenames):
        	filenames[file[0]] = file[1][:-4]
        return set(filenames)

#call this once to generate hearchal data
#os.system('bvh2csv -H '+source_dir+file+".bvh")
for file in getBvhNames():
	print(file)
	os.system('bvh-converter -r '+source_dir+file+".bvh")