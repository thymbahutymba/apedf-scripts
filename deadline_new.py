#!/usr/bin/env python3

import numpy as np
import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
import statistics


def plot(data, title, stdev = False):
    markers = [
        'o',
        '^',
        's',
        'd',
        'X',
        'h'
    ]

    plt.title(title, fontsize=24)
    plt.xlabel("Utilization", fontsize=22)
    plt.ylabel("deadline miss ratio", fontsize=22)
    plt.subplots_adjust(top = 0.95, bottom = 0.05, right = 0.95, left = 0.05, 
            hspace = 0, wspace = 0)
    matplotlib.rcParams.update({'font.size': 22})

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.yscale('log')
    plt.grid()

    for i, (k, d) in enumerate(data.items()):
        #linestyle='None',
        #plt.errorbar(d[:, 0], d[:, 1], yerr=np.transpose(d[:, [2,3]]), marker=markers[i], linestyle='None', elinewidth=2, label=k, markersize=16, capsize=5, capthick=3)
        #plt.errorbar(d[:, 0], d[:, 1], d[:, 2], marker=markers[i], linestyle='None', elinewidth=2, label=k, markersize=12, capsize=7, capthick=3)
        if not stdev:
            plt.plot(d[:, 0], d[:, 1], marker=markers[i], linewidth=3, label=k, markersize=16)
        else:
            plt.errorbar(d[:, 0] + (i - 1) * 0.01, d[:, 1], d[:, 2], marker=markers[i], linestyle='None', elinewidth=2, label=k, markersize=12, capsize=7, capthick=3)
    
    plt.legend(loc='upper left', fontsize=22)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, nargs=1, help="Path to logs")
    parser.add_argument("-t", type=str, required=True, help="Title of the plot")
    parser.add_argument("--stdev", action='store_true', help="Plot standard deviation")

    args = parser.parse_args()
    base_path = os.path.abspath(args.path[0])
    data = {}
    
    for policy in os.listdir(base_path):
        # Ignore possible images stored in the same directory
        if not os.path.isdir(os.path.join(base_path, policy)):
            continue

        data_mat = np.empty([0, 3])

        for util in os.listdir(os.path.join(base_path, policy)):
            missed_deadline = 0
            total_row = 0
            ratio = []

            for conf in os.listdir(os.path.join(base_path, policy, util)):
                md = 0
                tr = 0

                for file in os.listdir(os.path.join(base_path, policy, util, conf)):
                    with open(os.path.join(base_path, policy, util, conf, file), 'r') as f:
                        lines = f.readlines()[1:]

                    
                    md += sum([list(map(int, l.split()))[7] < 0 for l in lines])
                    tr += len(lines)
                    
                    #ratio = missed_deadline / total_row

                    #if ratio < min:
                    #    min = ratio
                    #elif ratio > max:
                    #    max = ratio
                    
                try:
                    ratio.append(md / tr)
                except:
                    print(os.path.join(base_path, policy, util, conf))

                missed_deadline += md
                total_row += tr

            #data_mat = np.vstack([data_mat, [float(util[:-1]), missed_deadline / total_row, statistics.stdev(ratio)]])
            data_mat = np.vstack([data_mat, [float(util[:-1]), statistics.mean(ratio), statistics.stdev(ratio)]])

        data[policy] = data_mat[np.argsort(data_mat[:, 0])]

    plot(data, args.t, args.stdev)


if __name__ == '__main__':
    main()
