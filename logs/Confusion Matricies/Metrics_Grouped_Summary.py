import glob
import numpy as np
from cf_matrix import *
import matplotlib.pyplot as plt


combos = ['n-n', 'n-o', 'n-s',
          's-s', 's-o', 's-n',
          'on-o', 'on-n', 'rn-n',
          'rn-rn', 'os-o', 'os-s',
          'rs-s', 'rs-rs','o-o',
          'Decision_Tree',
          'Random_Forest_Classifier',
          'Neural_Network',
          'Support_Vector_Machine',
          'K_Nearest_Neighbors']

_dict = {'n': "Normalized",
         's': "Standardized",
         'r': "Relitivized",
         'o': "Offset"}

for c in combos:
    summary = np.full((3, 3), 0.0)
    cm_list = [np.load(f) for f in glob.glob('Breakdown/*_'+c+'*npy')]+[np.load(f) for f in glob.glob('Breakdown/'+c+'*npy')]
    if len(cm_list)>0:
        processes = c.split('-')
        title = c
        if len(processes)==2:
            title = ''.join([_dict.get(p, '') for p in processes[0]])+" / "+''.join([_dict.get(p, '') for p in processes[1]])
        for cm in cm_list:
            summary += cm
        average = summary/len(cm_list)
        array = summary
        rows = np.sum(array,axis=1)
        columns = np.sum(array,axis=0)
        diagonal = np.diagonal(array)
        precision = 0
        recall = 0
        f_score = 0
        for i, n in enumerate(diagonal):
            i_precision = n / columns[i]
            i_recall = n / rows[i]
            i_f_score = 2*(i_precision*i_recall)/(i_precision+i_recall)
            precision += i_precision
            recall += i_recall
            f_score += i_f_score
            """print(i-1)
            print("\tPrecision:","{:.2f}".format(i_precision))
            print("\tRecall:","{:.2f}".format(i_recall))
            print("\tF-Score:","{:.2f}".format(i_f_score))"""
            
        print(c)
        print("\tPrecision:","{:.4f}".format(precision/3))
        print("\tRecall:","{:.4f}".format(recall/3))
        print("\tF-Score:","{:.4f}".format(f_score/3))
        print("\tAccuracy:","{:.2f}%".format(np.sum(diagonal)/np.sum(summary)*100))
    else:
        print(c, "----- Empty -----")
