__author__ = "Ami Abutbul"


# Full hands segmentation

# seg_colors = [
#     [0, 0, 0],       # background
#     [164, 164, 164], # person
#     [64, 128, 64],   # left thumb 1
#     [64, 128, 64],   # left thumb 2
#     [64, 128, 64],   # left thumb 3
#     [192, 0, 128],   # left index 1
#     [192, 0, 128],   # left index 2
#     [192, 0, 128],   # left index 3
#     [0, 128, 192],   # left middle 1
#     [0, 128, 192],   # left middle 2
#     [0, 128, 192],   # left middle 3
#     [128, 0, 0],     # left ring 1
#     [128, 0, 0],     # left ring 2
#     [128, 0, 0],     # left ring 3
#     [64, 0, 128],    # left little 1
#     [64, 0, 128],    # left little 2
#     [64, 0, 128],    # left little 3
#     [64, 0, 192],    # left palm
#     [64, 128, 64],   # right thumb 1
#     [64, 128, 64],   # right thumb 2
#     [64, 128, 64],   # right thumb 3
#     [192, 0, 128],   # right index 1
#     [192, 0, 128],   # right index 2
#     [192, 0, 128],   # right index 3
#     [0, 128, 192],   # right middle 1
#     [0, 128, 192],   # right middle 2
#     [0, 128, 192],   # right middle 3
#     [128, 0, 0],     # right ring 1
#     [128, 0, 0],     # right ring 2
#     [128, 0, 0],     # right ring 3
#     [64, 0, 128],    # right little 1
#     [64, 0, 128],    # right little 2
#     [64, 0, 128],    # right little 3
#     [64, 0, 192],    # right palm
#   ]

# Partial hands segmentation

seg_colors = [
    [0, 0, 0],       # background
    [164, 164, 164], # person
    [64, 128, 64],   # thumb
    [192, 0, 128],   # index
    [0, 128, 192],   # middle
    [128, 0, 0],     # ring
    [64, 0, 128],    # little
    [64, 0, 192],    # palm
  ]

full_to_partial_seg = {
    0: 0,
    1: 1,
    2: 2,
    3: 2,
    4: 2,
    5: 3,
    6: 3,
    7: 3,
    8: 4,
    9: 4,
    10: 4,
    11: 5,
    12: 5,
    13: 5,
    14: 6,
    15: 6,
    16: 6,
    17: 7,
    18: 2,
    19: 2,
    20: 2,
    21: 3,
    22: 3,
    23: 3,
    24: 4,
    25: 4,
    26: 4,
    27: 5,
    28: 5,
    29: 5,
    30: 6,
    31: 6,
    32: 6,
    33: 7
}
