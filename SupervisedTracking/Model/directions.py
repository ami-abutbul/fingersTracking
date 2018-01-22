import math as mt
import numpy as np

number_of_directions = 36
alpha = mt.pi / (number_of_directions / 2)

directions = np.array([[mt.cos(i * alpha), mt.sin(i * alpha)] for i in range(number_of_directions)])
