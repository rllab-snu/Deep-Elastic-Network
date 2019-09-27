import tensorflow as tf


BN_EPSILON = 0.001
weight_decay = 0.0001


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer()):

    regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

    new_variables = tf.get_variable(name=name, shape=shape, initializer=initializer, regularizer=regularizer)

    return new_variables


def output_layer(input_layer, num_labels):

    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels],
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h = tf.matmul(input_layer, fc_w) + fc_b

    return fc_h


def batch_normalization_layer(input_layer, dimension):

    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer


def conv_bn_relu_layer(input_layer, filter_shape, stride):

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = tf.nn.relu(bn_layer)

    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride):

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')

    return conv_layer


def residual_block(input_layer, output_channel, keep, lv=3, first_block=False):

    input_channel = input_layer.get_shape().as_list()[-1]

    output_channel = output_channel * 4

    if first_block:
        increase_dim = True
        stride = 1
    else:
        if input_channel * 2 == output_channel:
            increase_dim = True
            stride = 2
        elif input_channel == output_channel:
            increase_dim = False
            stride = 1
        else:
            raise ValueError('Residual Block Error')

    with tf.variable_scope('conv1_in_block'):
        if first_block:
            conv_list = []
            for i in xrange(lv):
                if i is 0:
                    r = int(output_channel / pow(2, lv - i - 1))
                else:
                    r = int(output_channel / pow(2, lv - i))
                with tf.variable_scope('hidden_lv%d' % i):
                    filter = create_variables('conv', [3, 3, input_channel, r])
                    conv1 = tf.sign(tf.nn.relu(keep - i)) * tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
                    conv_list.append(conv1)
            conv1 = tf.concat(conv_list, 3)
        else:
            conv_list = []
            for i in xrange(lv):
                if i is 0:
                    r = int(output_channel / pow(2, lv - i - 1))
                else:
                    r = int(output_channel / pow(2, lv - i))
                with tf.variable_scope('hidden_lv%d' % i):
                    conv1 = tf.sign(tf.nn.relu(keep - i)) * bn_relu_conv_layer(input_layer, [3, 3, input_channel, r], stride)
                    conv_list.append(conv1)
            conv1 = tf.concat(conv_list, 3)
    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

    if increase_dim is True:
        if first_block:
            padded_input = tf.pad(input_layer, [[0, 0], [0, 0], [0, 0], [24, 24]])
        else:
            pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                          input_channel // 2]])
    else:
        padded_input = input_layer
    output = conv2 + padded_input

    return output


def inference(input_batch, nclasses, n, id, keeps, reuse, input_reuse):

    j = 0
    layers = []
    with tf.variable_scope('widenet', reuse=input_reuse):
        with tf.variable_scope('conv0_size%d' % id, reuse=input_reuse):
            conv0 = conv_bn_relu_layer(input_batch, [7, 7, 3, 16], int(id/32))
            layers.append(conv0)

    with tf.variable_scope('widenet', reuse=reuse):
        for i in range(n):
            with tf.variable_scope('conv1_%d' % i, reuse=reuse):
                if i == 0:
                    conv1 = residual_block(layers[-1], 16, keeps[:, j, :, :, :], first_block=True)
                else:
                    conv1 = residual_block(layers[-1], 16, keeps[:, j, :, :, :])
                j += 1
                layers.append(conv1)

        for i in range(n):
            with tf.variable_scope('conv2_%d' % i, reuse=reuse):
                conv2 = residual_block(layers[-1], 32, keeps[:, j, :, :, :])
                j += 1
                layers.append(conv2)

        for i in range(n):
            with tf.variable_scope('conv3_%d' % i, reuse=reuse):
                conv3 = residual_block(layers[-1], 64, keeps[:, j, :, :, :])
                j += 1
                layers.append(conv3)
            assert conv3.get_shape().as_list()[1:] == [8, 8, 256]

    with tf.variable_scope('widenet', reuse=input_reuse):
        with tf.variable_scope('fc_size%d' % id, reuse=input_reuse):
            in_channel = layers[-1].get_shape().as_list()[-1]
            bn_layer = batch_normalization_layer(layers[-1], in_channel)
            relu_layer = tf.nn.relu(bn_layer)
            global_pool = tf.reduce_mean(relu_layer, [1, 2])

            assert global_pool.get_shape().as_list()[-1:] == [256]
            output = output_layer(global_pool, nclasses)
            layers.append(output)

    return layers[-1]


def policy_inference(input_batch, n, id, reuse, input_reuse):

    layers = []

    with tf.variable_scope('policy', reuse=input_reuse):
        with tf.variable_scope('conv0_size%d' % id, reuse=input_reuse):
            conv0 = conv_bn_relu_layer(input_batch, [7, 7, 3, 16], int(id/32))
            layers.append(conv0)

    with tf.variable_scope('policy', reuse=reuse):
        for i in range(1):
            with tf.variable_scope('conv1_%d' % i, reuse=reuse):
                if i == 0:
                    conv1 = residual_block(layers[-1], 16, tf.ones((1, 1, 1, 1)), first_block=True)
                else:
                    conv1 = residual_block(layers[-1], 16, tf.ones((1, 1, 1, 1)))
                layers.append(conv1)

        for i in range(1):
            with tf.variable_scope('conv2_%d' % i, reuse=reuse):
                conv2 = residual_block(layers[-1], 32, tf.ones((1, 1, 1, 1)))
                layers.append(conv2)

        for i in range(1):
            with tf.variable_scope('conv3_%d' % i, reuse=reuse):
                conv3 = residual_block(layers[-1], 64, tf.ones((1, 1, 1, 1)))
                layers.append(conv3)
            assert conv3.get_shape().as_list()[1:] == [8, 8, 256]

        with tf.variable_scope('fc', reuse=reuse):
            in_channel = layers[-1].get_shape().as_list()[-1]
            bn_layer = batch_normalization_layer(layers[-1], in_channel)
            relu_layer = tf.nn.relu(bn_layer)
            global_pool = tf.reduce_mean(relu_layer, [1, 2])

            assert global_pool.get_shape().as_list()[-1:] == [256]
            output = output_layer(global_pool, 12 * n)
            layers.append(output)

    return layers[-1]

