import math as mt
import numpy as np

def p2p_distance(x1, y1, x2, y2):
    return mt.sqrt(mt.pow((x1 - x2), 2) + mt.pow((y1 - y2), 2))


def in_circle(x, y, r, x1, y1):
    return p2p_distance(x, y, x1, y1) <= r


def get_point_in_distance(x0, y0, direction, distance, norm_dist=True):
    x_gag = direction.x
    y_gag = direction.y
    if norm_dist:
        alpha = distance
    else:
        alpha = mt.sqrt((mt.pow(distance, 2)) / (mt.pow(x_gag, 2) + mt.pow(y_gag, 2)))
    return int(round(alpha * x_gag + x0)), int(round(alpha * y_gag + y0))


def get_direction_index(directions, point):
    sub = directions - point
    square = np.square(sub)
    sum = np.sum(square, axis=1)
    return np.argmin(sum)
