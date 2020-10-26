#!/usr/bin/env python3

import numpy as np
import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
import sys
from matplotlib.lines import Line2D
from policy import Policy
import statistics


big_font = 24
small_font = 20


def multiple_plot(data_array, title, fname):
    markers = ['o', '^', 's', 'd', 'X', 'h', 'v', '*']
    fig, axs = plt.subplots(len(data_array), squeeze=False)
    axs = axs.flatten()
    matplotlib.rcParams.update({'font.size': big_font})

    fig.text(0.01, 0.5, 'Average migrations per taskset', va='center', rotation='vertical', fontsize=small_font)

    for j, data in enumerate(data_array):
        fig.suptitle(title, fontsize=big_font)

        axs[j].set_xlabel("Utilization", fontsize=small_font)
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


def log_parser(base_path):
    data = {}

    for policy in os.listdir(base_path):
        # Ignore possible images stored in the same directory
        if not os.path.isdir(os.path.join(base_path, policy)):
            continue

        data_mat = np.empty([0, 2], dtype=object)

        for util in os.listdir(os.path.join(base_path, policy)):
            ratio = []

            for conf in os.listdir(os.path.join(base_path, policy, util)):
                for file in os.listdir(
                        os.path.join(base_path, policy, util, conf)):
                    with open(
                            os.path.join(base_path, policy, util, conf, file),
                            'r') as f:
                        content = f.readlines()
                        nr_job = len(content[1:-1])
                        migrations = int(content[-1].split()[-1])

                    try:
                        ratio.append(migrations / nr_job)
                    except:
                        print(os.path.join(base_path, policy, util, conf))

            if any(ratio):
                data_mat = np.vstack([
                    data_mat,
                    np.array([
                        float(util[:-1]),
                        statistics.mean(ratio),
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

    args = parser.parse_args()
    base_path = os.path.abspath(args.path[0])

    if len(args.path) > 2:
        parser.error("Could not be specified more than 2 logs path.")
        sys.exit(1)

    data_array = []
    for i in range(0, len(args.path)):
        base_path = os.path.abspath(args.path[i])
        data_array.append(log_parser(base_path))

    multiple_plot(data_array, args.t, args.save)


if __name__ == '__main__':
    main()
