import matplotlib.pyplot as plt
import os
import cv2


def visualize_tradeoff(data, fields, axis_labels, optimal_curve=None, save_name=None):

    plt.rcParams["figure.figsize"] = (8,6)

    x_field = fields[0]
    y_field = fields[1]

    if optimal_curve is not None:
        plt.plot(optimal_curve[x_field], optimal_curve[y_field], label='Optimal Curve', c='black')
    
    for label in data:
        data_dict = data[label]
        plt.scatter(data_dict['data'][x_field], data_dict['data'][y_field], label=label,
                    marker=data_dict['marker'], c=data_dict['colour'], s=data_dict['size'])
    
    x_axis_label = axis_labels[0]
    y_axis_label = axis_labels[1]

    plt.xlabel(x_axis_label, fontdict={"size": 19})
    plt.ylabel(y_axis_label, fontdict={"size": 19})
    plt.tick_params(axis='both', which='major', labelsize=19)
    plt.legend(loc='upper left', prop={'size': 19}) #bbox_to_anchor=(1.01, 1.05),

    if save_name is not None:
        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
    else:
        plt.show()
    
    plt.clf()
    plt.rcParams.update(plt.rcParamsDefault)

