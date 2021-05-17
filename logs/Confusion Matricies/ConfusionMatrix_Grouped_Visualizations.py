import glob
import numpy as np
from cf_matrix import *
import matplotlib.pyplot as plt


"""combos = ['n-n', 'n-o', 'n-s',
          's-s', 's-o', 's-n',
          'on-o', 'on-n', 'rn-n',
          'rn-rn', 'os-o', 'os-s',
          'rs-s', 'rs-rs',
          'Decision_Tree',
          'Random_Forest_Classifier',
          'Neural_Network',
          'Support_Vector_Machine',
          'K_Nearest_Neighbors']"""

combos = ['rn-n']

_dict = {'n': "Normalized",
         's': "Standardized",
         'r': "Relitivized",
         'o': "Custom"}

cumulative = np.full((3, 3), 0.0)
#cumulative_title = "Top 5 Preprocessing Techniques"
cumulative_title = "(Location) Relativized, then Normalized;(Rotation) Normalized"
for c in combos:
    summary = np.full((3, 3), 0.0)
    cm_list = [np.load(f) for f in glob.glob('Breakdown/*_'+c+'*npy')]
    print(c, len(cm_list))
    processes = c.split('-')
    title = c
    if len(processes)==2:
        title = ''.join([_dict.get(p, '') for p in processes[0]])+" / "+''.join([_dict.get(p, '') for p in processes[1]])
    for cm in cm_list:
        summary += cm
        cumulative += cm
    average = summary/len(cm_list)
    categories = ["Negative", "Neutral", "Positive"]
    make_confusion_matrix(summary,
                            categories=categories,
                            title=title+"\n(Cumulative)")
    plt.savefig(title.replace('/','-')+" (Cumulative)"+'.png', dpi = 500, transparent=True)
    make_confusion_matrix(cumulative,
                            categories=categories,
                            title=cumulative_title+"\n(Cumulative)")
    plt.savefig(cumulative_title+".png", dpi = 500, transparent=True)
    make_confusion_matrix(average,
                            categories=categories,
                            title=title+"\n(Average)")
    plt.savefig(title.replace('/','-')+" (Average)"+'.png', dpi = 500, transparent=True)
