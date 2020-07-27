#!/usr/bin/env python3

"""
linestyle None
"""

import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def append(args):
    files_lines = [f.readlines()[1:] for f in args.file]
    missed_deadline = 0

    # Compute utilization
    U = 0
    for r in [fl[0] for fl in files_lines]:
        val = list(map(int, r.split()))
        U += (val[8] * 100 / 90) / val[9]

    print("U: " + str(round(U, 1)))

    # Sum all deadline miss between all tasks
    for lines in files_lines:
        missed_deadline += sum(
            [list(map(int, l.split()))[7] < 0 for l in lines])

    md_percentage = missed_deadline / sum([len(l) for l in files_lines])
    
    if md_percentage:
        args.s.write(args.l + "," + str(round(U, 1)) + "," + str(md_percentage) + '\n')

def plot(args):
    lines = args.i.readlines()

    markers = [
        '-o',
        '-^',
        '-s',
        '-d',
        '-X',
        '-h'
    ]

    labels = []
    U = []
    missed_deadline = []
    i = -1
    for l in lines:
        [lab, u, md] = l.strip('\n').split(',')

        if i == -1 or lab != labels[i]:
            i += 1
            labels.append(lab)
            U.append([])
            missed_deadline.append([])

        U[i].append(float(u)) 
        missed_deadline[i].append(float(md))

    plt.title(args.t, fontsize=24)
    plt.xlabel("Utilization", fontsize=22)
    plt.ylabel("Percentage of missed deadlines", fontsize=22)
    plt.subplots_adjust(top = 0.95, bottom = 0.05, right = 0.95, left = 0.05, 
            hspace = 0, wspace = 0)
    matplotlib.rcParams.update({'font.size': 22})

    for i in range(0, len(U)):
        plt.plot(U[i], missed_deadline[i], markers[i], label=labels[i], linewidth=3, markersize=16)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.legend(labels, loc='upper left', fontsize=22)
    plt.grid()

    plt.yscale('log')

    plt.show()


def main():
    parser = argparse.ArgumentParser()

    sp = parser.add_subparsers()
    sp_append = sp.add_parser(
        'append', help='Add element to list that contains missed deadline')
    sp_append.set_defaults(func=append)
    sp_append.add_argument('file', type=argparse.FileType('r'), nargs='+')
    sp_append.add_argument('-l', type=str, required=True, help='Label used later for plotting')
    sp_append.add_argument('-s', type=argparse.FileType('a'), required=True, help='File for store intermediate data before plotting')

    sp_plot = sp.add_parser('plot', help='Create the final plot')
    sp_plot.add_argument('-i', type=argparse.FileType('r'), required=True, help='Load data for plotting from file')
    sp_plot.add_argument('-t', type=str, required=True, help="Title of the plot")
    sp_plot.set_defaults(func=plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
