import matplotlib.pyplot as plt
import numpy as np



def plot(x, y, xlabel,ylabel, xlim, ylim, lenlabel,title, fig, file_fig):
#    plt.figure(fig, figsize=(5, 2))
#     plt.title(title, fontsize=20, y=1.08)
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    colors = ['r', 'b', 'g', 'k', 'm', 'y', 'c']
    cs = [ele + '-' for ele in colors]
    # cs = ['ro-', 'g^-', 'bx-', 'kd-', 'ms-', '']
    dot = ['o', '^', '*', 'd', 's', '*']
    if len(y) > 1:
        for i in range(len(y)):
            plt.plot(x, y[i], cs[i], label=lenlabel[i])
            plt.scatter(x, y[i], c='c', s=15, marker=dot[i])
        # plt.legend(bbox_to_anchor=(0, 0.26, 0.96, 1), bbox_transform=plt.gcf().transFigure, loc=4, fontsize=6.5)
        plt.legend(borderpad=2, bbox_transform=plt.gcf().transFigure, loc=1, fontsize=20)
        # leg = plt.gca().get_legend()
        # leg.draw_frame(False)
        # ltext = leg.get_texts()  # all the text.Text instance in the legend
        # llines = leg.get_lines()  # all the lines.Line2D instance in the legend
        # plt.setp(ltext, fontsize=6)  # the legend text fontsize
        # plt.setp(llines, linewidth=2)  # the legend linewidth
    else:
        plt.plot(x, y[0], '-o')
    # plt.axes().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(file_fig)


def plotBar(y, width, xlabel, ylabel, title, filename):
    N = len(y)
    ind = np.arange(N)  # the x locations for the groups
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, y , width, color='r')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(xlabel)
    for rect in rects1:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height, '%d'%int(height), ha='center', va='bottom')
    plt.savefig(filename)