import matplotlib.pyplot as plt
from Model.configuration import *
import numpy as np
import os
import sys

RESOLUTION = [1600, 900]#[GetSystemMetrics(0) - 1, GetSystemMetrics(1) - 1]  # [Width, Height]


def file_to_list(file):
    res = []
    with open(file, "r") as file:
        for line in file:
            res.append(float(line))
            # res += list(map(lambda x: float(x), list(filter(lambda x: x != '', line.split(",")))))
    return res

if __name__ == '__main__':
    # if len(sys.argv) == 1:
    #     print("Error: missing run mode type (hand/fingers)")
    #     sys.exit(1)
    #
    # run_mode_type = sys.argv[1]
    # set_run_config(run_mode_type)

    loss = file_to_list(os.path.join("stats", loss_file))

    plt.figure(figsize=(RESOLUTION[0]/96, RESOLUTION[1]/96), dpi=96)

    sub1 = plt.subplot(111)
    sub1.set_title("rewards")
    sub1.plot(np.arange(1, len(loss)+1), loss)

    plt.show()

