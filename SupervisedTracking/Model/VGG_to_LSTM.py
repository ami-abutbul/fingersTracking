__author__ = "Ami Abutbul"
import tensorflow as tf
import numpy as np
from Model.configuration import *
from Utilities.nn_utils import conv_relu, conv_2_half_size, fully_connected
from Utilities.file_utils import create_dir
from Model.Study import StudiesHandler
import tensorflow.contrib.slim as slim


class Tracker(object):
    def __init__(self):
        self.input_frames = tf.placeholder(tf.float32, shape=[batch_size, image_height, image_width, 3], name='input')
        self.cell_state = tf.placeholder(tf.float32, [batch_size, rnn_state_size])
        self.hidden_state = tf.placeholder(tf.float32, [batch_size, rnn_state_size])
        self.target = tf.placeholder(tf.float32, [batch_size, 2])
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
                out, _ = conv_relu(out, 512, 3, "conv5_3")
                out = tf.nn.max_pool(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

                out, _ = conv_2_half_size(out, 512, "conv6_1")  # to reduce image spatial size

            with tf.variable_scope("preRNN"):
                flat = slim.flatten(out)  # TODO: check that this tensor have shape of [batch_size, flatten(out)]
                out = tf.nn.dropout(flat, keep_prob=self.keep_prob)
                out, _ = fully_connected(out, rnn_state_size, "fc7")
                out = tf.reshape(out, shape=[batch_size, 1, -1])
                self.check = out

            with tf.variable_scope("RNN"):
                state = tf.nn.rnn_cell.LSTMStateTuple(self.cell_state, self.hidden_state)
                cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_state_size, state_is_tuple=True)
                out, self.out_state = tf.nn.dynamic_rnn(cell, out, initial_state=state, dtype=tf.float32)
                out = tf.reshape(out, shape=[batch_size, -1])

            with tf.variable_scope("prediction"):
                self.predict, _ = fully_connected(out, 2, "fc8")
                self.loss = tf.reduce_mean(tf.losses.absolute_difference(labels=self.target, predictions=self.predict))
                trained_vars = [v for v in tf.trainable_variables() if "FingersTracker" in v.name]
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, var_list=trained_vars)
        return self


def train(studies_dir, list_of_dirs=False):
    tf.reset_default_graph()

    tracker = Tracker().build()
    studies_handler = StudiesHandler(studies_dir, list_of_dirs=list_of_dirs)

    create_dir(checkpoint_dir)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    keep_prob = 0.5
    update_prob_rate = epoch_num*0.95 / 10

    with tf.Session() as sess:
        sess.run(init)

        if restore_model:
            print('Loading Model...')
            saver.restore(sess, checkpoint_file)

        for i in range(epoch_num):
            if i == update_prob_rate:
                update_prob_rate *= 2
                keep_prob += 0.05

            study = studies_handler.get_study()
            print("{}) study: {}".format(i, study.study_dir))

            current_cell_state = np.zeros((batch_size, rnn_state_size))
            current_hidden_state = np.zeros((batch_size, rnn_state_size))

            warm_frames = study.get_warm_frames()
            for frame in warm_frames:
                next_state = sess.run(tracker.out_state, feed_dict={tracker.input_frames: [frame],
                                                                      tracker.keep_prob: keep_prob,
                                                                      tracker.cell_state: current_cell_state,
                                                                      tracker.hidden_state: current_hidden_state})
                current_cell_state, current_hidden_state = next_state

            while not study.is_end_of_study():
                image, point = study.next()
                point = [point.x, point.y]
                _, loss, next_state = sess.run([tracker.optimizer, tracker.loss, tracker.out_state],
                # f = sess.run(tracker.check,
                                         feed_dict={tracker.input_frames: image,
                                                    tracker.target: [point],
                                                    tracker.keep_prob: keep_prob,
                                                    tracker.cell_state: current_cell_state,
                                                    tracker.hidden_state: current_hidden_state})
                # f = np.array(f)
                # print(f.shape)
                current_cell_state, current_hidden_state = next_state
                print(loss)

            saver.save(sess, checkpoint_file)


if __name__ == '__main__':
    if mode == "train":
        if platform.system() == 'Linux':
            train("/home/ami/handsTrack/studies/fingers/20.12.17")

        else:
            train("C:/Users/il115552/Desktop/New folder (6)")