import matplotlib.pyplot as plt
import numpy as np
import os
from cf_matrix import *

_dict = {'n': "Normalized",
         's': "Standardized",
         'r': "Relitivized",
         'o': "Offset"}

for cm_file in os.listdir():
    if "ConfusionMatrix" in cm_file:
        print(cm_file)
        model = cm_file[0:cm_file.find('_ConfusionMatrix')].replace('_', ' ')
        processes = cm_file[cm_file.rfind('_')+1:-4].split('-')
        title=model+"\n("+''.join([_dict[p] for p in processes[0]])+" / "+''.join([_dict[p] for p in processes[1]])+")"
        cm = np.load(cm_file)
        categories = ["Negative", "Neutral", "Positive"]
        make_confusion_matrix(cm,
                              categories=categories,
                              title=title)
        plt.savefig(title.replace('\n', ' - ').replace('/','-')+'.png', dpi = 500, transparent=True)



