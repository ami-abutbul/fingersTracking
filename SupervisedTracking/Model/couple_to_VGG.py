__author__ = "Ami Abutbul"
import tensorflow as tf
import numpy as np
from Model.configuration import *
from Utilities.nn_utils import conv_relu, conv_2_half_size, fully_connected
from Utilities.file_utils import create_dir
from Utilities.stdout_log import StdoutLog
from Model.Study import StudiesHandler
import tensorflow.contrib.slim as slim
import sys


class Tracker(object):
    def __init__(self):
        self.minibatch_size = tf.placeholder(tf.int32, shape=[], name='minibatch_size')
        self.input_frames = tf.placeholder(tf.float32, shape=[None, image_height, image_width, 3], name='input')
        self.target = tf.placeholder(tf.float32, [None, 2])
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        self.loss = None
        self.out_state = None
        self.optimizer = None
        self.predict = None
        self.check = None

    def build(self):
        with tf.variable_scope("FingersTracker"):
            with tf.variable_scope("VGG16"):
                out, _ = conv_relu(self.input_frames, 64, 3, "conv1_1")
                out, _ = conv_relu(out, 64, 3, "conv1_2")
                out = tf.nn.max_pool(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

                out, _ = conv_relu(out, 128, 3, "conv2_1")
                out, _ = conv_relu(out, 128, 3, "conv2_2")
                out = tf.nn.max_pool(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

                out, _ = conv_relu(out, 256, 3, "conv3_1")
                out, _ = conv_relu(out, 256, 3, "conv3_2")
                out, _ = conv_relu(out, 256, 3, "conv3_3")
                out = tf.nn.max_pool(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

                out, _ = conv_relu(out, 512, 3, "conv4_1")
                out, _ = conv_relu(out, 512, 3, "conv4_2")
                out, _ = conv_relu(out, 512, 3, "conv4_3")
                out = tf.nn.max_pool(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

                out, _ = conv_relu(out, 512, 3, "conv5_1")
                out, _ = conv_relu(out, 512, 3, "conv5_2")
                out, _ = conv_2_half_size(out, 512, "conv5_3")  # to reduce image spatial size
                out = tf.nn.max_pool(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

            with tf.variable_scope("fullyConnected"):
                out = slim.flatten(out)
                out = tf.nn.dropout(out, keep_prob=self.keep_prob)
                out, _ = fully_connected(out, main_fc_size, "fc7")
                out = tf.nn.relu(out)
                out = tf.reshape(out, [self.minibatch_size, main_fc_size*2])
                self.predict, _ = fully_connected(out, 2, "fc8")

            with tf.variable_scope("optimizer"):
                self.loss = tf.reduce_sum(tf.square(self.predict - self.target))
                #self.optimizer = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=learning_rate).minimize(self.loss)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        return self


def train(studies_dir, list_of_dirs=False):
    tf.reset_default_graph()

    tracker = Tracker().build()
    studies_handler = StudiesHandler(studies_dir, list_of_dirs=list_of_dirs)

    create_dir(checkpoint_dir)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    keep_prob = 0.8
    # update_prob_rate = epoch_num*0.9 / 6

    with tf.Session() as sess:
        sess.run(init)

        if restore_model:
            print('Loading Model...')
            saver.restore(sess, checkpoint_file)

        for i in range(epoch_num):
            # if i == update_prob_rate:
            #     update_prob_rate *= 2
            #     keep_prob += 0.05

            study = studies_handler.get_study()
            print("\n#####################################################")
            print("{}) study: {}".format(i, study.study_dir))
            print("#####################################################")

            while not study.is_end_of_study():
                minibatch_size = 0
                images = []
                points = []
                while not study.is_end_of_study() and minibatch_size < batch_size:
                    image1, image2, point = study.next_couple()
                    if image2 is None:
                        break
                    minibatch_size += 1
                    images.append(image1.reshape((image_height, image_width, 3)))
                    images.append(image2.reshape((image_height, image_width, 3)))
                    points.append([point.x, point.y])

                if len(images) == 0:
                    continue

                _, loss, p = sess.run([tracker.optimizer, tracker.loss, tracker.predict],
                                         feed_dict={tracker.minibatch_size: minibatch_size,
                                                    tracker.input_frames: images,
                                                    tracker.target: points,
                                                    tracker.keep_prob: keep_prob})
                p = np.array(p)
                print("------------------------------------------------------------------")
                print("loss = {}".format(loss))
                for i in range(len(points)):
                    print("predict: {}, relative: {}".format(p[i], points[i]))

            saver.save(sess, checkpoint_file)


def test(studies_dir, list_of_dirs=False):
    tf.reset_default_graph()

    tracker = Tracker().build()
    studies_handler = StudiesHandler(studies_dir, list_of_dirs=list_of_dirs)

    create_dir(checkpoint_dir)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    keep_prob = 1

    with tf.Session() as sess:
        sess.run(init)
        print('Loading Model...')
        saver.restore(sess, checkpoint_file)

        while not studies_handler.epoch_done:

            study = studies_handler.get_study()
            print("\n#####################################################")
            print("study: {}".format(study.study_dir))
            print("#####################################################")

            current_cell_state = np.zeros((batch_size, rnn_state_size))
            current_hidden_state = np.zeros((batch_size, rnn_state_size))

            warm_frames = study.get_warm_frames()
            for frame in warm_frames:
                next_state = sess.run(tracker.out_state, feed_dict={tracker.input_frames: [frame],
                                                                      tracker.keep_prob: keep_prob,
                                                                      tracker.cell_state: current_cell_state,
                                                                      tracker.hidden_state: current_hidden_state})
                current_cell_state, current_hidden_state = next_state

            old_state = current_cell_state
            while not study.is_end_of_study():
                image, point = study.next()
                point = [point.x, point.y]
                loss, next_state, p, check = sess.run([tracker.loss, tracker.out_state, tracker.predict, tracker.check],
                                         feed_dict={tracker.input_frames: image,
                                                    tracker.target: [point],
                                                    tracker.keep_prob: keep_prob,
                                                    tracker.cell_state: current_cell_state,
                                                    tracker.hidden_state: current_hidden_state})
                current_cell_state, current_hidden_state = next_state

                # print("predict: {}, relative: {}, loss = {}".format(np.array(p), point, loss))
                # print(np.mean(current_cell_state - old_state))
                # old_state = current_cell_state

if __name__ == '__main__':
    sys.stdout = StdoutLog(log_file, sys.stdout, print_to_log, print_to_stdout)

    if mode == "train":
        print("Start training ..")
        if platform.system() == 'Linux':
            train("/home/ami/fingersTracking/data/try")  
            #train(["/home/ami/fingersTracking/data/studies",
            #      "/home/ami/fingersTracking/data/studies9.1.18"], list_of_dirs=True)
        else:
            train("C:/Users/il115552/Desktop/New folder (6)")

    if mode == "test":
        print("Start testing ..")
        if platform.system() == 'Linux':
            test("/home/ami/fingersTracking/data/test")
        else:
            test("C:/Users/il115552/Desktop/New folder (6)")
