import tensorflow as tf


BN_EPSILON = 0.00001
weight_decay = 0.0001


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer()):

    regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

    new_variables = tf.get_variable(name=name, shape=shape, initializer=initializer, regularizer=regularizer)

    return new_variables


def output_layer(input_layer, num_labels):

    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels],
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    with tf.variable_scope('conv3_%d' % i, reuse=reuse):
        conv3 = residual_block(layers[-1], 256, keeps[:, j, :, :, :])
        j += 1
        layers.append(conv3)

        for i in range(n):
            with tf.variable_scope('conv4_%d' % i, reuse=reuse):
                conv4 = residual_block(layers[-1], 512, keeps[:, j, :, :, :])
                j += 1
                layers.append(conv4)
            assert conv4.get_shape().as_list()[1:] == [4, 4, 512]

    with tf.variable_scope('resnet', reuse=input_reuse):
        with tf.variable_scope('fc_size%d' % id, reuse=input_reuse):
            in_channel = layers[-1].get_shape().as_list()[-1]
            bn_layer = batch_normalization_layer(layers[-1], in_channel)
            relu_layer = tf.nn.relu(bn_layer)
            global_pool = tf.reduce_mean(relu_layer, [1, 2])

            assert global_pool.get_shape().as_list()[-1:] == [512]
            output = output_layer(global_pool, nclasses)
            layers.append(output)

    return layers[-1]


def policy_inference(input_batch, n, id, reuse, input_reuse):

    layers = []

    with tf.variable_scope('policy', reuse=input_reuse):
        with tf.variable_scope('conv0_size%d' % id, reuse=input_reuse):
            conv0 = conv_bn_relu_layer(input_batch, [7, 7, 3, 64], int(id/32))
            layers.append(conv0)

    with tf.variable_scope('policy', reuse=reuse):
        for i in range(1):
            with tf.variable_scope('conv1_%d' % i, reuse=reuse):
                if i == 0:
                    conv1 = residual_block(layers[-1], 64, tf.ones((1, 1, 1, 1)), first_block=True)
                else:
                    conv1 = residual_block(layers[-1], 64, tf.ones((1, 1, 1, 1)))
                layers.append(conv1)

        for i in range(1):
            with tf.variable_scope('conv2_%d' % i, reuse=reuse):
                conv2 = residual_block(layers[-1], 128, tf.ones((1, 1, 1, 1)))
                layers.append(conv2)

        for i in range(1):
            with tf.variable_scope('conv3_%d' % i, reuse=reuse):
                conv3 = residual_block(layers[-1], 256, tf.ones((1, 1, 1, 1)))
                layers.append(conv3)

        for i in range(1):
            with tf.variable_scope('conv4_%d' % i, reuse=reuse):
                conv4 = residual_block(layers[-1], 512, tf.ones((1, 1, 1, 1)))
                layers.append(conv4)
            assert conv4.get_shape().as_list()[1:] == [4, 4, 512]

        with tf.variable_scope('fc', reuse=reuse):
            in_channel = layers[-1].get_shape().as_list()[-1]
            bn_layer = batch_normalization_layer(layers[-1], in_channel)
            relu_layer = tf.nn.relu(bn_layer)
            global_pool = tf.reduce_mean(relu_layer, [1, 2])

            assert global_pool.get_shape().as_list()[-1:] == [512]
            output = output_layer(global_pool, 16 * n)
            layers.append(output)

    return layers[-1]

