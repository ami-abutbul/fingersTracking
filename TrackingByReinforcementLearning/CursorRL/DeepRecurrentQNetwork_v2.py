import sys
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from HandSeg.HandSegNet import encoder
from Utilities.cnn_utils import conv_2_half_size
from Utilities.file_utils import create_dir, get_file_name
from Utilities.stdout_log import StdoutLog
from CursorRL.configuration import *
from CursorRL.Environment import Environment
from CursorRL.ExperiencedStudiesBuffer import ExperiencedStudiesBuffer


class DRQN(object):
    def __init__(self, lstm_cell, scope_name, reuse_encoder_variables, input_frames=None):
        self.trace_len = tf.placeholder(dtype=tf.int32, name=scope_name+'trace_len')
        self.input_frames = input_frames if input_frames is not None else tf.placeholder(tf.float32, shape=[None, image_height, image_width, 3], name=scope_name+'input')
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name=scope_name+'batch_size')
        self.target_Q = tf.placeholder(shape=[None], dtype=tf.float32, name=scope_name+'target_Q')
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name=scope_name+'actions')
        self.reuse_encoder_variables = reuse_encoder_variables
        self.lstm_cell = lstm_cell
        self.scope_name = scope_name
        self.state_in = None
        self.rnn_state = None
        self.predict = None
        self.optimizer = None
        self.Qout = None
        self.loss = None
        self.out_encoder = None

    def build_graph(self):
        # out: [mb_size, image_size/16, image_size/16, 1024]
        self.out_encoder, _, encoder_learned_variables = encoder(self.input_frames, self.reuse_encoder_variables)

        with tf.variable_scope(self.scope_name):
            # out: [image_size/32, image_size/32, 512]
            conv_layer, conv_vars = conv_2_half_size(self.out_encoder, 512, scope_name=self.scope_name + 'down_sampling1')
            out = tf.nn.relu(conv_layer, name=self.scope_name + 'down_sampling_relu1')

            # out: [image_size/64, image_size/64, 512]
            pool_out = tf.nn.max_pool(out, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')

            # out: [image_size/128, image_size/128, 256]
            conv_layer, conv_vars = conv_2_half_size(pool_out, 256, scope_name=self.scope_name + 'down_sampling2')
            out = tf.nn.relu(conv_layer, name=self.scope_name + 'down_sampling_relu2')

            # out: [image_size/256, image_size/256, 256 => 2,1,256]
            conv_layer, conv_vars = conv_2_half_size(out, 256, scope_name=self.scope_name + 'down_sampling3')
            out = tf.nn.relu(conv_layer, name=self.scope_name + 'down_sampling_relu3')

            # Take the output from the final conv layer and send it to a recurrent layer.
            # The input reshaped into [batch x trace x units] for rnn processing,
            # and then returned to [batch x units].
            conv_flat = tf.reshape(slim.flatten(out), [self.batch_size, self.trace_len, units_size])
            self.state_in = self.lstm_cell.zero_state(self.batch_size, tf.float32)
            rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=conv_flat, cell=self.lstm_cell, dtype=tf.float32,
                                                    initial_state=self.state_in, scope=self.scope_name + '_lstm')

            flat_rnn = tf.reshape(rnn, shape=[-1, units_size])
            fc_weights = tf.get_variable(self.scope_name + '_fc_weights', [units_size, actions_num],
                                         initializer=tf.contrib.layers.xavier_initializer())
            fc_bias = tf.get_variable(self.scope_name + '_fc_bias', [actions_num],
                                      initializer=tf.zeros_initializer())
            self.Qout = tf.matmul(flat_rnn, fc_weights) + fc_bias

            self.predict = tf.argmax(self.Qout, 1)

            actions_one_hot = tf.one_hot(self.actions, actions_num, dtype=tf.float32)
            Q = tf.reduce_sum(tf.multiply(tf.nn.softmax(self.Qout), actions_one_hot))
            # t_Q = tf.multiply(tf.nn.softmax(self.target_Q), actions_one_hot)
            td_error = tf.square(self.target_Q - Q)

            self.loss = tf.reduce_mean(td_error)*1e3
            trained_vars = [v for v in tf.trainable_variables() if self.scope_name in v.name]
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, var_list=trained_vars)
        return self, trained_vars

    @classmethod
    def update_target_graph(cls, tf_vars, update_ratio):
        main_var = []
        target_var = []
        op_holder = []

        for idx, var in enumerate(tf_vars):
            if 'main_DRQN' in var.name:
                main_var.append(var)
            elif 'target_DRQN' in var.name:
                target_var.append(var)

        for idx in range(len(target_var)):
            op_holder.append(target_var[idx].assign((main_var[idx].value() * update_ratio) +
                                                    ((1 - update_ratio) * target_var[idx].value())))
        return op_holder

    @classmethod
    def update_target(cls, op_holder, sess):
        for op in op_holder:
            sess.run(op)

    @classmethod
    def load_encoder_weights(cls):
        variables = tf.trainable_variables()
        encoder_vars = [v for v in variables if "Encoder" in v.name]
        return tf.train.Saver(encoder_vars)

    @classmethod
    def warm_lstm_state(cls, sess, net, current_lstm_state, warm_frames):
        return sess.run(net.rnn_state, feed_dict={net.input_frames: warm_frames,
                                                  net.trace_len: len(warm_frames),
                                                  net.state_in: current_lstm_state,
                                                  net.batch_size: 1})


def train(studies_dir, list_of_dirs=False):
    tf.reset_default_graph()

    # We define the cells for the main and target q-networks
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=units_size, state_is_tuple=True)
    cell_t = tf.contrib.rnn.BasicLSTMCell(num_units=units_size, state_is_tuple=True)
    main_net, main_net_vars = DRQN(cell, 'main_DRQN', reuse_encoder_variables=False).build_graph()
    target_net, target_net_vars = DRQN(cell_t, 'target_DRQN', reuse_encoder_variables=True).build_graph()

    init = tf.global_variables_initializer()
    encoder_loader = DRQN.load_encoder_weights()
    saver = tf.train.Saver(main_net_vars + target_net_vars)
    create_dir(checkpoint_dir)
    target_ops = DRQN.update_target_graph(tf.trainable_variables(), update_target_ratio)

    experienced_studies_buffer = ExperiencedStudiesBuffer(500)
    environment = Environment(studies_dir, stats_dir, list_of_dirs=list_of_dirs)

    # Set the rate of random action decrease.
    e = startE
    step_drop = (startE - endE) / annealing_steps
    total_steps = 0

    with tf.Session() as sess:
        sess.run(init)
        encoder_loader.restore(sess, encoder_checkpoint_file)

        if restore_model:
            print('Loading Model...')
            saver.restore(sess, checkpoint_file)

        DRQN.update_target(target_ops, sess)  # Set the target network to be equal to the main network.
        for i in range(epoch_num):
            print("study number: {}".format(i))
            # Reset environment and get warm frames
            warm_frames = environment.reset()

            # Warm up the recurrent layer's hidden state
            lstm_state = (np.zeros([1, units_size]), np.zeros([1, units_size]))
            lstm_state = DRQN.warm_lstm_state(sess, main_net, lstm_state, warm_frames)

            # Get first state
            s = environment.start()
            j = 0

            # The Q-Network
            while j < max_study_len:
                j += 1
                # Choose an action randomly from the Q-network
                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    lstm_state1 = sess.run(main_net.rnn_state, feed_dict={main_net.input_frames: s,
                                                                          main_net.trace_len: 1,
                                                                          main_net.state_in: lstm_state,
                                                                          main_net.batch_size: 1})
                    a = np.random.randint(0, actions_num)
                else:
                    a, lstm_state1 = sess.run([main_net.predict, main_net.rnn_state],
                                              feed_dict={main_net.input_frames: s,
                                                         main_net.trace_len: 1,
                                                         main_net.state_in: lstm_state,
                                                         main_net.batch_size: 1})
                    a = a[0]

                s1, r, d = environment.step(a)
                total_steps += 1
                if total_steps > pre_train_steps:
                    if e > endE:
                        e -= step_drop

                    if total_steps % update_freq == 0:
                        DRQN.update_target(target_ops, sess)

                        experienced_studies_buffer.select_study()
                        warm_frames = experienced_studies_buffer.get_warm_frames()
                        done_trace = False

                        # Reset the recurrent layer's hidden state
                        state_train = (np.zeros([mb_size, units_size]), np.zeros([mb_size, units_size]))
                        main_state_train = DRQN.warm_lstm_state(sess, main_net, state_train, warm_frames)
                        target_state_train = DRQN.warm_lstm_state(sess, target_net, state_train, warm_frames)

                        while not done_trace:
                            # train_batch contains list of [s, a, r, s1, d]
                            train_batch, trace_length, done_trace = experienced_studies_buffer.get_trace()
                            if train_batch is None:
                                break

                            # update the Double-DQN to the target Q-values
                            main_next_state, Q1 = sess.run([main_net.rnn_state, main_net.predict],
                                                            feed_dict={main_net.input_frames: np.vstack(train_batch[:, 3]),
                                                                       main_net.trace_len: trace_length,
                                                                       main_net.state_in: main_state_train,
                                                                       main_net.batch_size: mb_size})

                            target_next_state, Q2 = sess.run([target_net.rnn_state, target_net.Qout],
                                                             feed_dict={target_net.input_frames: np.vstack(train_batch[:, 3]),
                                                                        target_net.trace_len: trace_length,
                                                                        target_net.state_in: target_state_train,
                                                                        target_net.batch_size: mb_size})

                            # end_multiplier = -(train_batch[:, 4] - 1)
                            # doubleQ = Q2[range(mb_size * trace_length), Q1]
                            targetQ = train_batch[:, 2] # + (gamma * doubleQ * end_multiplier)
                            # Update the network with our target values.
                            _, loss = sess.run([main_net.optimizer, main_net.loss], feed_dict={main_net.input_frames: np.vstack(train_batch[:, 0]),
                                                                                               main_net.target_Q: targetQ,
                                                                                               main_net.actions: train_batch[:, 1],
                                                                                               main_net.trace_len: trace_length,
                                                                                               main_net.state_in: state_train,
                                                                                               main_net.batch_size: mb_size})
                            print("len: {},\tloss: {}".format(trace_length, loss))
                            main_state_train = main_next_state
                            target_state_train = target_next_state
                s = s1
                lstm_state = lstm_state1
                if d:
                    break

            experienced_studies_buffer.append(environment.current_study)

            if i % 100 == 0 and i != 0 and i > pre_train_steps:
                saver.save(sess, checkpoint_file)

        saver.save(sess, checkpoint_file)


def test(studies_dir):
    tf.reset_default_graph()

    # We define the cells for the main and target q-networks
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=units_size, state_is_tuple=True)
    main_net, main_net_vars = DRQN(cell, 'main_DRQN', reuse_encoder_variables=False).build_graph()

    init = tf.global_variables_initializer()
    encoder_loader = DRQN.load_encoder_weights()
    q_net_loader = tf.train.Saver(main_net_vars)

    environment = Environment(studies_dir, stats_dir)

    with tf.Session() as sess:
        sess.run(init)
        encoder_loader.restore(sess, encoder_checkpoint_file)
        q_net_loader.restore(sess, checkpoint_file)

        num_of_studies = len(environment.studies)
        environment.WRITE_STATS_STEPS = 2*num_of_studies  # write results at the end

        for i in range(num_of_studies):
            warm_frames = environment.reset()

            # Warm up the recurrent layer's hidden state
            lstm_state = (np.zeros([1, units_size]), np.zeros([1, units_size]))
            lstm_state = DRQN.warm_lstm_state(sess, main_net, lstm_state, warm_frames)

            # Get first state
            s = environment.start()
            done = False

            while not done:
                a, lstm_state = sess.run([main_net.predict, main_net.rnn_state],
                                          feed_dict={main_net.input_frames: s,
                                                     main_net.trace_len: 1,
                                                     main_net.state_in: lstm_state,
                                                     main_net.batch_size: 1})
                s, _, done = environment.step(a[0])

                study_name = get_file_name(environment.current_study.study_dir)
            path_len = environment.current_study.image_index - warm_frames_num
            print("{}:\t len {},\t done: {}".format(study_name, path_len, environment.current_study.finished_successfully))

        environment.write_stats()


if __name__ == '__main__':
    # if len(sys.argv) == 1:
    #     print("Error: missing run mode type (hand/fingers)")
    #     sys.exit(1)
    #
    # run_mode_type = sys.argv[1]
    # set_run_config(run_mode_type)

    sys.stdout = StdoutLog(log_dir, sys.stdout, print_to_log, print_to_stdout)
    create_dir(checkpoint_dir)

    if mode == "train":
        if platform.system() == 'Linux':
            if mode_type == "hand":
                print("use hands dataset..")
                train(["/home/ami/handsTrack/studies/full_hands/17.12.17",
                       "/home/ami/handsTrack/studies/full_hands/18.12.17",
                       "/home/ami/handsTrack/studies/full_hands/19.12.17"],
                      list_of_dirs=True)
            else:
                print("use fingers dataset..")
                train("/home/ami/handsTrack/studies/fingers/20.12.17")

        else:
            if mode_type == "hand":
                print("use hands dataset..")
                train(["D:/private/datasets/handTrack/studies/full_hands/17.12.17",
                       "D:/private/datasets/handTrack/studies/full_hands/18.12.17",
                       "D:/private/datasets/handTrack/studies/full_hands/19.12.17"],
                      list_of_dirs=True)
            else:
                print("use fingers dataset..")
                train("D:/private/datasets/handTrack/studies/fingers/20.12.17")

    else:
        if platform.system() == 'Linux':
            if mode_type == "hand":
                print("use hands test set..")
                test("/home/ami/handsTrack/studies/test_studies/full_hands")
            else:
                print("use fingers test set..")
                test("/home/ami/handsTrack/studies/test_studies/fingers")

        else:
            if mode_type == "hand":
                print("use hands test set..")
                test("D:/private/datasets/handTrack/test_studies/full_hands")
            else:
                print("use fingers test set..")
                test("D:/private/datasets/handTrack/test_studies/fingers")
