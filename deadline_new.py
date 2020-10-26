#!/usr/bin/env python3

import numpy as np
import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
import statistics
import sys
from matplotlib.lines import Line2D
from policy import Policy


big_font = 24
small_font = 20


def multiple_plot(data_array, title, fname, stdev=False, scatter=False):
    markers = ['o', '^', 's', 'd', 'X', 'h', 'v', '*']
    fig, axs = plt.subplots(len(data_array), squeeze=False)
    axs = axs.flatten()
    matplotlib.rcParams.update({'font.size': big_font})

    for j, data in enumerate(data_array):
        fig.suptitle(title, fontsize=big_font)

        axs[j].set_xlabel("Utilization", fontsize=small_font)
        axs[j].set_ylabel("deadline miss ratio", fontsize=small_font)

#        ax = axs[j].twinx()
#        ax.set_ylabel("with invariance" if not j else "without invariance",
#                      fontsize=small_font,
#                      rotation=-90,
#                      labelpad=30)
#
#        ax.set_yticklabels([])
#        ax.set_yticks([])
        axs[j].set_xticks(
            np.arange(
                min(np.array(list(data.values())[0])[:, 0]), 
                max(np.array(list(data.values())[0])[:, 0]) + 0.1,
                0.1
            )
        )
        
        axs[j].set_yscale('log')
        axs[j].grid()

        for i, (k, d) in enumerate(data.items()):
            if stdev:
                axs[j].errorbar(d[:, 0] + (i - 1) * 0.01,
                                d[:, 1],
                                d[:, 2],
                                marker=markers[Policy.from_str(k).value],
                                color='C' + str(Policy.from_str(k).value),
                                linestyle='None',
                                elinewidth=2,
                                label=k,
                                markersize=12,
                                capsize=7,
                                capthick=3)
            elif scatter:
                for ue, re in zip(d[:, 0], d[:, 3]):
                    axs[j].scatter(
                        [ue + (i - 1) * 0.02] * len(re),
                        re,
                        s=40,
                        marker=markers[Policy.from_str(k).value],
                        color='C' + str(Policy.from_str(k).value),
                    )
                axs[j].plot(d[:, 0] + (i - 1) * 0.02,
                            d[:, 1],
                            marker=markers[Policy.from_str(k).value],
                            color='C' + str(Policy.from_str(k).value),
                            linestyle='None',
                            label=k.replace("a2p", "a$^2$p").replace("edf", "EDF").replace("wf", "WF").replace("ff", "FF"),
                            markersize=12)
            else:
                axs[j].plot(d[:, 0],
                            d[:, 1],
                            marker=markers[Policy.from_str(k).value],
                            color='C' + str(Policy.from_str(k).value),
                            linewidth=1.5,
                            label=k.replace("a2p", "a$^2$p").replace("edf", "EDF").replace("wf", "WF").replace("ff", "FF"),
                            markersize=12)
                axs[j].plot(d[d[:, 4] != 0, 0],
                            d[d[:, 4] != 0, 4],
                            marker=markers[Policy.from_str(k).value],
                            color='C' + str(Policy.from_str(k).value),
                            linewidth=1.5,
                            linestyle='dashed',
                            markersize=12)

        handles, labels = np.array(axs[j].get_legend_handles_labels(),
                                   dtype=object)
        order = [Policy.from_str(l).value for l in labels]

        median = Line2D([], [],
                        color="black",
                        linewidth=1.5,
                        linestyle="--",
                        label='median')

        if not scatter and not stdev:
            axs[j].legend(np.append(handles[np.argsort(order)], median),
                          np.append(labels[np.argsort(order)], "median"),
                          loc='upper left',
                          fontsize=small_font)
        else:
            axs[j].legend(handles[np.argsort(order)],
                          labels[np.argsort(order)],
                          loc='upper left',
                          fontsize=small_font)

        axs[j].tick_params(axis='both', labelsize=20)

    #plt.subplots_adjust(
    #    top=0.95,
    #    bottom=0.05,
    #    right=0.95,
    #    left=0.05,
    #    hspace=0.005,
    #    wspace=0)
    plt.subplots_adjust(
        top=0.94,
        bottom=0.08,
        right=0.97,
        left=0.07,
        hspace=0.005,
        wspace=0)
    #plt.subplots_adjust(
    #    top=0.94,
    #    bottom=0.07,
    #    right=0.96,
    #    left=0.08,
    #    hspace=0.005,
    #    wspace=0)

    fig = plt.gcf()
    #fig.set_size_inches((3840 / 100., 2160 / 100.))
    #fig.set_size_inches((1920 / 100., 1080 / 100.))
    #fig.set_size_inches((1500 / 100., 1200 / 100.))
    fig.set_size_inches((1900 / 100., 850 / 100.))


    if fname is None:
        plt.show()
    else:
        fig.savefig(fname)


def log_parser(base_path):
    data = {}

    for policy in os.listdir(base_path):
        # Ignore possible images stored in the same directory
        if not os.path.isdir(os.path.join(base_path, policy)):
            continue

        data_mat = np.empty([0, 5], dtype=object)

        for util in os.listdir(os.path.join(base_path, policy)):
            missed_deadline = 0
            total_row = 0
            ratio = []

            for conf in os.listdir(os.path.join(base_path, policy, util)):
                md = 0
                tr = 0

                for file in os.listdir(
                        os.path.join(base_path, policy, util, conf)):
                    with open(
                            os.path.join(base_path, policy, util, conf, file),
                            'r') as f:
                        lines = f.readlines()[1:-1]

                    md += sum(
                        [list(map(int, l.split()))[7] < 0 for l in lines])
                    tr += len(lines)

                try:
                    ratio.append(md / tr)
                except:
                    print(os.path.join(base_path, policy, util, conf))

                missed_deadline += md
                total_row += tr

            if any(ratio):
                data_mat = np.vstack([
                    data_mat,
                    np.array([
                        float(util[:-1]),
                        statistics.mean(ratio),
                        statistics.stdev(ratio),
                        ratio,
                        statistics.median(ratio),
                    ],
                    dtype=object)
                ])

        data[policy] = data_mat[np.argsort(data_mat[:, 0])]

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, nargs='+', help="Path to logs")
    parser.add_argument("-t",
                        type=str,
                        required=True,
                        help="Title of the plot")
    parser.add_argument("--save",
                        nargs='?',
                        type=argparse.FileType('wb'),
                        help="Save figure to file instead of showing it")

    command_group = parser.add_mutually_exclusive_group()
    command_group.add_argument("--stdev",
                               action='store_true',
                               help="Standard deviation of all the run")
    command_group.add_argument("--scatter",
                               action='store_true',
                               help="Scatter plot of deadline miss ration")

    args = parser.parse_args()
    base_path = os.path.abspath(args.path[0])

    if len(args.path) > 2:
        parser.error("Could not be specified more than 2 logs path.")
        sys.exit(1)

    data_array = []
    for i in range(0, len(args.path)):
        base_path = os.path.abspath(args.path[i])
        data_array.append(log_parser(base_path))

    multiple_plot(data_array, args.t, args.save, args.stdev, args.scatter)


if __name__ == '__main__':
    main()
