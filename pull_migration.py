#!/usr/bin/env python3

import numpy as np
import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
import sys
from matplotlib.lines import Line2D
from policy import Policy


big_font = 24
small_font = 20


def multiple_plot(data_array, title, fname):
    markers = ['o', '^', 's', 'd', 'X', 'h', 'v', '*']
    fig, axs = plt.subplots(len(data_array), squeeze=False)
    axs = axs.flatten()
    matplotlib.rcParams.update({'font.size': big_font})

    fig.text(0.01, 0.5, 'Average pull migrations per taskset', va='center', rotation='vertical', fontsize=small_font)
    #plt.ylabel("Average pull migrations per taskset", fontsize=small_font)

    for j, data in enumerate(data_array):
        fig.suptitle(title, fontsize=big_font)

        axs[j].set_xlabel("Utilization", fontsize=small_font)
        #axs[j].set_ylabel("Average pull migrations per taskset", fontsize=small_font)

#        ax = axs[j].twinx()
#        ax.set_ylabel("with invariance" if not j else "without invariance",
#                      fontsize=small_font,
#                      rotation=-90,
#                      labelpad=30)
#        ax.set_yticklabels([])
#        ax.set_yticks([])

        axs[j].set_xticks(
            np.arange(
                min(np.array(list(data.values())[0])[:, 0]), 
                max(np.array(list(data.values())[0])[:, 0]) + 0.1,
                0.1
            )
        )

        axs[j].grid()

        for (k, d) in data.items():
            axs[j].plot(d[:, 0],
                        d[:, 1],
                        marker=markers[Policy.from_str(k).value],
                        color='C' + str(Policy.from_str(k).value),
                        linewidth=1.5,
                        label=k.replace("a2p", "a$^2$p").replace("edf", "EDF").replace("wf", "WF").replace("ff", "FF"),
                        markersize=12)

        handles, labels = np.array(axs[j].get_legend_handles_labels(),
                                   dtype=object)
        order = [Policy.from_str(l).value for l in labels]

        axs[j].legend(handles[np.argsort(order)],
                      labels[np.argsort(order)],
                      loc='upper right',
                      fontsize=small_font)

        axs[j].tick_params(axis='both', labelsize=20)

    plt.subplots_adjust(
        top=0.94,
        bottom=0.08,
        right=0.97,
        left=0.07,
        hspace=0.005,
        wspace=0)

    fig = plt.gcf()
    fig.set_size_inches((1900 / 100., 850 / 100.))

    if fname is None:
        plt.show()
    else:
        fig.savefig(fname)


def log_parser(file):
    data = {}
    policy = ""

    data_mat = np.empty([0, 2])

    for line in file.readlines():
        content = line.split()
        content[1:] = list(map(float, content[1:]))

        if not policy:
            policy = content[0]

        if policy != content[0]:
            data[policy] = data_mat
            policy = content[0]
            data_mat = np.empty([0, 2])

        if content[3] >= content[2]:
            migration = content[3] - content[2]
        else:
            migration = 2**32 - 1 - content[2] + content[3]

        data_mat = np.vstack([data_mat, [content[1], migration / 30]])

    data[policy] = data_mat
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file",
                        type=argparse.FileType('r'),
                        nargs='+',
                        help="Log files")
    parser.add_argument("-t",
                        type=str,
                        required=True,
                        help="Title of the plot")
    parser.add_argument("--save",
                        nargs='?',
                        type=argparse.FileType('wb'),
                        help="Save figure to file instead of showing it")

    args = parser.parse_args()

    if len(args.file) > 2:
        parser.error("Could not be specified more than 2 log files.")
        sys.exit(1)

    data_array = []
    for f in args.file:
        data_array.append(log_parser(f))

    multiple_plot(data_array, args.t, args.save)


if __name__ == '__main__':
    main()
