from lib.general_dependencies import *

def _get_unique_figure_labels(figure):
    f_handles = []
    f_labels = []
    for axis in figure.get_axes():
        a_handles, a_labels = axis.get_legend_handles_labels()
        f_handles+=a_handles
        f_labels+=a_labels
    by_label = dict(zip(f_labels, f_handles))
    return by_label

def _get_unique_axis_labels(axis):
    handles, labels = axis.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    return by_label

def _get_plot_points(metric_item):
    if isinstance(metric_item, list):
        return metric_item
    elif isinstance(metric_item, dict):
        return list(metric_item.values())
    else:
        return [metric_item]

def plot_history_individual(history_dict, title, base_label, graphics_dir, validation_label='validation', save_figure=False):
    # summarize history for loss
    for metric in [m for m in history_dict.keys() if 'val' not in m]:
        plt.plot(_get_plot_points(history_dict[metric]), label=base_label)
        plt.plot(_get_plot_points(history_dict.get('val_'+metric, [])), label=validation_label)
        plt.xlabel('Epoch')
        y_label = ' '.join(i.capitalize() for i in metric.split('_'))
        plt.ylabel(y_label)
        plt.legend(loc='upper right', fontsize='x-large')
        plt.title(title)
        if(save_figure):
            plt.savefig(os.path.join(graphics_dir, 'generator/training_history_'+metric+'.png'), dpi = 500, transparent=True)
        plt.clf()
        
def _plot_group(metric_list, dictionary, title, savefile):
    for metric in metric_list:
        plt.plot(_get_plot_points(dictionary[metric]), label=metric)
        plt.plot(_get_plot_points(dictionary.get('val_'+metric, [])), label='val_'+metric if dictionary.get('val_'+metric) else '')
    by_label = _get_unique_axis_labels(plt.gca())
    plt.legend(by_label.values(), by_label.keys(), ncol=2, loc='upper right', fontsize='x-small')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend(loc='upper right', ncol=2, fontsize='x-small')
    plt.title(title)
    if(savefile):
        plt.savefig(savefile, dpi = 500, transparent=True)
    plt.clf()
    
def plot_history_grouped(history_dict, title, graphics_dir, save_figures=False):
    savefile = None
    if(save_figures):
        savefile=os.path.join(graphics_dir, 'generator/training_history_group_{}.png')
    _plot_group([m for m in history_dict.keys() if 'val' not in m][1:], history_dict, title.format(1), savefile.format(1)) #'generator/_group_1_training_history.png'
    _plot_group([m for m in history_dict.keys() if 'val' not in m and 'divergence' not in m][1:], history_dict, title.format(2), savefile.format(2)) 
    _plot_group([m for m in history_dict.keys() if 'val' not in m and 'divergence' not in m and 'hinge' not in m][1:], history_dict, title.format(3), savefile.format(3))  
    
def _bar_group(x_ticks_labels, dictionary):
    fig = plt.figure()
    axis = fig.add_axes([0,0,1,1])
    for metric in x_ticks_labels:
        axis.bar(metric, _get_plot_points(dictionary[metric]))
    axis.set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=18)
    plt.clf()
    
def bar_history_grouped(history_dict, title, base_label, validation_label='validation'):
    _bar_group([m for m in history_dict.keys() if 'val' not in m], history_dict)
    _bar_group([m for m in history_dict.keys() if 'val' not in m and 'divergence' not in m], history_dict)
    _bar_group([m for m in history_dict.keys() if 'val' not in m and 'divergence' not in m and 'hinge' not in m], history_dict)