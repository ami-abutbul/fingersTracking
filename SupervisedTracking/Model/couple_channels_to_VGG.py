__author__ = "Ami Abutbul"
import tensorflow as tf
import numpy as np
from Model.configuration import *
from Utilities.nn_utils import conv_bn_relu, fully_connected, fully_connected_bn
from Utilities.file_utils import create_dir, write_list_to_file
from Utilities.stdout_log import StdoutLog
from Model.Study import StudiesHandler
import tensorflow.contrib.slim as slim
import sys


class Tracker(object):
    def __init__(self, is_training=False):
        self.minibatch_size = tf.placeholder(tf.int32, shape=[], name='minibatch_size')
        self.input_frames = tf.placeholder(tf.float32, shape=[None, image_height, image_width, input_channels*2], name='input')
        self.target = tf.placeholder(tf.float32, [None, 2])
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        self.is_training = is_training
        self.loss = None
        self.out_state = None
        self.optimizer = None
        self.predict = None
        self.check = None

    def build(self):
        with tf.variable_scope("FingersTracker"):
            with tf.variable_scope("VGG16_conv"):
                out = tf.nn.avg_pool(self.input_frames, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name='avg_pool')

                out, _ = conv_bn_relu(out, 64, 3, self.is_training, "conv1_1")
                out, _ = conv_bn_relu(out, 64, 3, self.is_training, "conv1_2")
                out = tf.nn.max_pool(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

                out, _ = conv_bn_relu(out, 128, 3, self.is_training, "conv2_1")
                out, _ = conv_bn_relu(out, 128, 3, self.is_training, "conv2_2")
                out = tf.nn.max_pool(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

                out, _ = conv_bn_relu(out, 256, 3, self.is_training, "conv3_1")
                out, _ = conv_bn_relu(out, 256, 3, self.is_training, "conv3_2")
                out, _ = conv_bn_relu(out, 256, 3, self.is_training, "conv3_3")
                out = tf.nn.max_pool(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

                out, _ = conv_bn_relu(out, 512, 3, self.is_training, "conv4_1")
                out, _ = conv_bn_relu(out, 512, 3, self.is_training, "conv4_2")
                out, _ = conv_bn_relu(out, 512, 3, self.is_training, "conv4_3")
                out = tf.nn.max_pool(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

                out, _ = conv_bn_relu(out, 512, 3, self.is_training, "conv5_1")
                out, _ = conv_bn_relu(out, 512, 3, self.is_training, "conv5_2")
                out, _ = conv_bn_relu(out, 512, 3, self.is_training, "conv5_3")
                out = tf.nn.max_pool(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

            with tf.variable_scope("fullyConnected"):
                out = slim.flatten(out)

                # out = tf.nn.dropout(out, keep_prob=self.keep_prob)
                out, _ = fully_connected_bn(out, main_fc_size, self.is_training, "fc6")
                out = tf.nn.relu(out)

                out, _ = fully_connected_bn(out, main_fc_size, self.is_training, "fc7")
                out = tf.nn.relu(out)

                self.predict, _ = fully_connected(out, 2, "fc8")

            with tf.variable_scope("optimizer"):
                self.loss = tf.reduce_sum(tf.square(self.predict - self.target))
                # self.check = [v for v in tf.trainable_variables() if "fc8" in v.name and "weights" in v.name]
                # self.optimizer = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=learning_rate).minimize(self.loss)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        return self


def train(studies_dir, list_of_dirs=False):
    tf.reset_default_graph()

    tracker = Tracker(is_training=True).build()
    studies_handler = StudiesHandler(studies_dir, list_of_dirs=list_of_dirs)

    create_dir(checkpoint_dir)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    keep_prob = 1

    with tf.Session() as sess:
        sess.run(init)

        if restore_model:
            print('Loading Model...')
            saver.restore(sess, checkpoint_file)

        for i in range(epoch_num):
            loss_array = []
            study = studies_handler.get_study()
            print("\n#####################################################")
            print("{}) study: {}".format(i, study.study_dir))
            print("#####################################################")

            while not study.is_end_of_study():
                minibatch_size = 0
                images = []
                targets = []
                while not study.is_end_of_study() and minibatch_size < batch_size:
                    image1, image2, point = study.next_couple()
                    if image2 is None:
                        break
                    minibatch_size += 1
                    image1 = image1.reshape((image_height, image_width, input_channels))
                    image2 = image2.reshape((image_height, image_width, input_channels))
                    images.append(np.concatenate((image1, image2), axis=2))
                    targets.append([point.x, point.y])

                if len(images) == 0:
                    continue

                _, loss, predict = sess.run([tracker.optimizer, tracker.loss, tracker.predict],
                                         feed_dict={tracker.minibatch_size: minibatch_size,
                                                    tracker.input_frames: images,
                                                    tracker.target: targets,
                                                    tracker.keep_prob: keep_prob})
                print("------------------------------------------------------------------")
                print("loss = {}".format(loss))
                loss_array.append(loss)
                for j in range(len(targets)):
                    print("predict: {}, relative: {}".format(predict[j], targets[j]))

            write_list_to_file(loss_array, loss_file)
            if i % 100 == 0:
                saver.save(sess, checkpoint_file)


# def test(studies_dir, list_of_dirs=False):
#     tf.reset_default_graph()
#
#     tracker = Tracker().build()
#     studies_handler = StudiesHandler(studies_dir, list_of_dirs=list_of_dirs)
#
#     create_dir(checkpoint_dir)
#     saver = tf.train.Saver()
#     init = tf.global_variables_initializer()
#
#     keep_prob = 1
#
#     with tf.Session() as sess:
#         sess.run(init)
#         print('Loading Model...')
#         saver.restore(sess, checkpoint_file)
#
#         while not studies_handler.epoch_done:
#
#             study = studies_handler.get_study()
#             print("\n#####################################################")
#             print("study: {}".format(study.study_dir))
#             print("#####################################################")
#
#             current_cell_state = np.zeros((batch_size, rnn_state_size))
#             current_hidden_state = np.zeros((batch_size, rnn_state_size))
#
#             warm_frames = study.get_warm_frames()
#             for frame in warm_frames:
#                 next_state = sess.run(tracker.out_state, feed_dict={tracker.input_frames: [frame],
#                                                                       tracker.keep_prob: keep_prob,
#                                                                       tracker.cell_state: current_cell_state,
#                                                                       tracker.hidden_state: current_hidden_state})
#                 current_cell_state, current_hidden_state = next_state
#
#             old_state = current_cell_state
#             while not study.is_end_of_study():
#                 image, point = study.next()
#                 point = [point.x, point.y]
#                 loss, next_state, p, check = sess.run([tracker.loss, tracker.out_state, tracker.predict, tracker.check],
#                                          feed_dict={tracker.input_frames: image,
#                                                     tracker.target: [point],
#                                                     tracker.keep_prob: keep_prob,
#                                                     tracker.cell_state: current_cell_state,
#                                                     tracker.hidden_state: current_hidden_state})
#                 current_cell_state, current_hidden_state = next_state
#
#                 # print("predict: {}, relative: {}, loss = {}".format(np.array(p), point, loss))
#                 # print(np.mean(current_cell_state - old_state))
#                 # old_state = current_cell_state

if __name__ == '__main__':
    sys.stdout = StdoutLog(log_file, sys.stdout, print_to_log, print_to_stdout)
    print("input channels: {}".format(input_channels))
    if mode == "train":
        print("Start training ..")
        if platform.system() == 'Linux':
            # train(["/home/ami/fingersTracking/data/studies",
            #        "/home/ami/fingersTracking/data/studies9.1.18"], list_of_dirs=True)
            train("/home/ami/fingersTracking/data/try")
        else:
            train("C:/Users/il115552/Desktop/New folder (6)")

    # if mode == "test":
    #     print("Start testing ..")
    #     if platform.system() == 'Linux':
    #         test("/home/ami/fingersTracking/data/test")
    #     else:
    #         test("C:/Users/il115552/Desktop/New folder (6)")
