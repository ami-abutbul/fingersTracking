import math as mt

number_of_directions = 36
alpha = mt.pi / (number_of_directions / 2)

none_action = 0
click_action = 1
directions = [[mt.cos(i * alpha), mt.sin(i * alpha)] for i in range(number_of_directions)]
