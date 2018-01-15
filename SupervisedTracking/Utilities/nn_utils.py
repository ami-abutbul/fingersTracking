import tensorflow as tf


def initializer():
    # return tf.random_normal_initializer(1.0, 0.02)
    return tf.contrib.layers.xavier_initializer()


def pre_processing(img):
    return img / 255.


def leaky_relu(x, alpha=0.2):
    # this block looks like it has 2 inputs on the graph unless we do this
    x = tf.identity(x)

    return tf.maximum(x, alpha * x)


def batchnorm(batch_input, scope_name="batchnorm"):
    with tf.variable_scope(scope_name):
        # this block looks like it has 3 inputs on the graph unless we do this
        batch_input = tf.identity(batch_input)

        channels = batch_input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(batch_input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        res = tf.nn.batch_normalization(batch_input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return res, [offset, scale]


def conv(batch_input, out_channels, filter_size, scope_name="conv"):
    with tf.variable_scope(scope_name):
        in_channels = batch_input.get_shape()[3]
        filters = tf.get_variable("filter", [filter_size, filter_size, in_channels, out_channels], dtype=tf.float32, initializer=initializer())
        bias = tf.get_variable('bias', [out_channels], initializer=tf.zeros_initializer())
        conv_res = tf.nn.conv2d(batch_input, filters, [1, 1, 1, 1], padding="SAME")
        return conv_res + bias, [filters, bias]


def conv_relu(batch_input, out_channels, filter_size, scope_name="conv"):
    with tf.variable_scope(scope_name):
        in_channels = batch_input.get_shape()[3]
        filters = tf.get_variable("filter", [filter_size, filter_size, in_channels, out_channels], dtype=tf.float32, initializer=initializer())
        bias = tf.get_variable('bias', [out_channels], initializer=tf.zeros_initializer())
        conv_res = tf.nn.conv2d(batch_input, filters, [1, 1, 1, 1], padding="SAME")
        bias_conv = tf.nn.bias_add(conv_res, bias)
        relu = tf.nn.relu(bias_conv, name='relu')
        return relu, [filters, bias]


def conv_2_half_size(batch_input, out_channels, scope_name="conv_2_half_size"):
    with tf.variable_scope(scope_name):
        in_channels = batch_input.get_shape()[3]
        filters = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=initializer())
        bias = tf.get_variable('bias', [out_channels], initializer=tf.zeros_initializer())
        # padding the input:     turns to:
        # 1 1                    0 0 0 0
        # 1 1                    0 1 1 0
        #                        0 1 1 0
        #                        0 0 0 0
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv_res = tf.nn.conv2d(padded_input, filters, [1, 2, 2, 1], padding="VALID")
        return conv_res + bias, [filters, bias]


def deconv(batch_input, out_channels, filter_size, scope_name="deconv"):
    with tf.variable_scope(scope_name):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filters = tf.get_variable("filter", [filter_size, filter_size, out_channels, in_channels], dtype=tf.float32, initializer=initializer())
        bias = tf.get_variable('bias', [out_channels], initializer=tf.zeros_initializer())
        deconv_res = tf.nn.conv2d_transpose(batch_input, filters, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return deconv_res + bias, [filters, bias]


def fully_connected(batch_input, output_size, scope_name="fc"):
    with tf.variable_scope(scope_name):
        in_size = batch_input.get_shape()[1]  # [batch_size, input_size]
        weights = tf.get_variable('weights', [in_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', [output_size], initializer=tf.zeros_initializer())
        return tf.nn.bias_add(tf.matmul(batch_input, weights), bias), [weights, bias]
