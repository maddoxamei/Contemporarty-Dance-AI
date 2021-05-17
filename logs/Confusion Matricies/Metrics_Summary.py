import os
import numpy as np
from cf_matrix import *
import matplotlib.pyplot as plt

_dict = {'n': "Normalized",
         's': "Standardized",
         'r': "Relitivized",
         'o': "Other"}

for cm_file in [f for f in os.listdir('Breakdown') if '.npy' in f]:
    print(cm_file[0:cm_file.find('_ConfusionMatrix')].replace('_', ' '), cm_file[cm_file.rfind('_')+1:-4])
    array = np.load('Breakdown/'+cm_file)

    #print(array)
    
    rows = np.sum(array,axis=1)
    columns = np.sum(array,axis=0)
    diagonal = np.diagonal(array)
    precision = 0
    recall = 0
    f_score = 0
    for i, n in enumerate(diagonal):
        i_precision = (n / columns[i] if columns[i] else 0)
        i_recall = (n / rows[i] if rows[i] else 0)
        i_f_score = 2*(i_precision*i_recall)/(i_precision+i_recall if i_precision+i_recall else 1)
        precision += i_precision
        recall += i_recall
        f_score += i_f_score

    
    print("\tPrecision:","{:.4f}".format(precision/3))
    print("\tRecall:","{:.4f}".format(recall/3))
    print("\tF-Score:","{:.4f}".format(f_score/3))
    print("\tAccuracy:","{:.4f}".format(np.sum(diagonal)/np.sum(array)))
