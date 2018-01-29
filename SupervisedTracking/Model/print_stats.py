import matplotlib.pyplot as plt
from Model.configuration import *
import numpy as np
from Utilities.file_utils import write_list_to_file
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
    # res = []
    # with open("stats/loss.log", "r") as file:
    #     for line in file:
    #         res.append(float(line))
    #
    # res = np.array(res)
    # avg = []
    # for i in range(len(res)//64):
    #     j = 64*i
    #     sum_i = np.sum(res[j:j+64])
    #     avg.append(sum_i/64)
    #
    # write_list_to_file(avg, "stats/avg.log")


    loss = file_to_list(os.path.join("stats", "avg.log"))
    # loss = file_to_list(os.path.join("stats", loss_file))
    plt.figure(figsize=(RESOLUTION[0]/96, RESOLUTION[1]/96), dpi=96)

    sub1 = plt.subplot(111)
    sub1.set_title("loss")
    sub1.plot(np.arange(1, len(loss)+1), loss)
    plt.grid(True)
    plt.show()

