#!/usr/bin/env python3

import numpy as np
import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
import statistics
import sys
from matplotlib.lines import Line2D


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
    parser.add_argument("path", type=str, help="Path to logs")
    parser.add_argument("--save",
                        required=True,
                        type=argparse.FileType('w'),
                        help="Save to CSV file")

    args = parser.parse_args()
    data = log_parser(os.path.abspath(args.path))

    for (k, d) in data.items():
        args.save.write("# " + k + ' #\n')
        args.save.write("#util" + " " * 8 + "#mean" + " " * 8 + "#median\n")

        np.savetxt(args.save, d[:, [0,1,4]], fmt='%2.10f')

    args.save.close()

if __name__ == '__main__':
    main()
