import os, sys

source_dir = "../Data/BVH/"

def getFileNames():
        filenames = [f for f in os.listdir(source_dir) if f.endswith('.bvh')]
        for file in enumerate(filenames):
        	filenames[file[0]] = file[1][:-4]
        return set(filenames)


def rename():
	for file in getFileNames():
		original = source_dir+file+".bvh"
		
		filename = file.split('-')
		if('BVH' in filename):
			filename.remove('BVH')
		filename = '-'.join(filename)
		altered = source_dir+filename+".bvh"
		print(original)
		print(altered)
		#os.rename(original, altered)

def alter():
	for file in getFileNames():
		original = source_dir+file+".bvh"
		altered = source_dir+'_'.join(file.split('-'))+".bvh"
		print(original)
		print(altered)
		#os.rename(original, altered)
rename()
alter()