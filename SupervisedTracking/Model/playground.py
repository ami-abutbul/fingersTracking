__author__ = "Ami Abutbul"
from Model.Study import StudiesHandler
from Utilities.image_utils import mat_to_image
from Utilities.two_dimension_utils import get_direction_index
from Model.directions import directions
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    studiesHandler = StudiesHandler("C:/Users/il115552/Desktop/New folder (6)")
    study = studiesHandler.get_study()
    print("study.is_end_of_study: {}".format(study.is_end_of_study()))
    while not study.is_end_of_study():
        image1, image2, point = study.next_couple()
        if image2 is None:
            break
        image1 = image1.reshape((480, 640, 3))
        image2 = image2.reshape((480, 640, 3))
        print(np.concatenate((image1, image2), axis=2))
        print("-------------------------------------")


    # x = tf.constant([[[[1, 3, 5, 7], [1 ,1, 1, 1], [1 ,1, 1, 1], [1 ,1, 1, 1]],
    #                  [[2, 2, 2, 2], [2 ,2, 2, 2], [2 ,2, 2, 2], [2 ,2, 2, 2]],
    #                  [[3, 3, 3, 3], [3 ,3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]]]], dtype='float32')
    # x = tf.transpose(x, perm=[0, 2, 3, 1])
    # # x = tf.reshape(x, shape=[4,4,3])
    # x = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    # # x = tf.one_hot(x, 8, on_value=1.0, off_value=0.0, axis=-1)
    # with tf.Session() as sess:
    #     res = sess.run(x)
    #     print(res)
    #     print(np.array(res).shape)



    # x = -15
    # y = 0
    # point_np = np.array([x, y])
    # point_np = point_np / np.sqrt(x ** 2 + y ** 2)
    # print(get_direction_index(directions, point_np))