from CursorRL.DeepRecurrentQNetwork import DRQN
from CursorRL.configuration import *
from CursorRL import actions
import tensorflow as tf
import threading
import cv2
import numpy as np
import pyautogui


def build_DRQN():
    tf.reset_default_graph()

    frame = tf.placeholder(tf.float32, shape=[1, image_height, image_width, 3])
    input_queue = tf.FIFOQueue(capacity=30, dtypes=tf.float32, shapes=[1, image_height, image_width, 3])
    enqueue_op = input_queue.enqueue(frame)

    # We define the cells for the main and target q-networks
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=units_size, state_is_tuple=True)
    drq_net, drq_net_vars = DRQN(cell, 'main_DRQN', reuse_encoder_variables=False, input_frames=input_queue.dequeue()).build_graph()

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    DRQN.load_encoder_weights().restore(sess, encoder_checkpoint_file)
    tf.train.Saver(drq_net_vars).restore(sess, checkpoint_file)

    return sess, drq_net, enqueue_op, frame


def load_and_enqueue_frames(sess, enqueue_op, frame):
    cap = cv2.VideoCapture(0)
    while True:
        _, new_frame = cap.read()
        sess.run(enqueue_op, feed_dict={frame: np.expand_dims(new_frame, axis=0)})


def start_hand_controlled():
    sess, drq_net, enqueue_op, frame = build_DRQN()

    # Start a thread to enqueue data asynchronously, and hide I/O latency.
    t = threading.Thread(target=load_and_enqueue_frames, args=(sess, enqueue_op, frame))
    t.start()

    lstm_state = (np.zeros([1, units_size]), np.zeros([1, units_size]))

    while True:
        a, lstm_state = sess.run([drq_net.predict, drq_net.rnn_state],
                                 feed_dict={drq_net.trace_len: 1,
                                            drq_net.state_in: lstm_state,
                                            drq_net.batch_size: 1})
        action = a[0]

        if action == actions.none_action:
            continue
        elif action == actions.click_action:
            pyautogui.click()
        else:
            direction = actions.directions[action - 2]
            pyautogui.moveRel(int(round(step_size * direction[0])), int(round(step_size * direction[1])))


if __name__ == '__main__':
    start_hand_controlled()