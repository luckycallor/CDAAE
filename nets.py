import tensorflow as tf
import tensorflow.contrib.slim as slim

def Encoder_hot(images, image_size, hot_size, reuse=False, is_training=True, name='Encoder_hot'):
    # image_size = 32
    init_size = int(image_size/8)
    with tf.variable_scope(name, reuse=reuse):
        with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=None, stride=2,
                            weights_initializer=tf.contrib.layers.xavier_initializer()):
            with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                                activation_fn=tf.nn.relu, is_training=is_training):
                net = slim.conv2d(images, 64, [3, 3], scope='conv1')
                # [batch_size, image_size/2, image_size/2, 64]
                net = slim.batch_norm(net, scope='bn1')
                net = slim.conv2d(net, 128, [3, 3], scope='conv2')
                # [batch_size, image_size/4, image_size/4, 128]
                mid = slim.batch_norm(net, scope='bn2')
                net = slim.conv2d(mid, 256, [3, 3], scope='conv3')
                # [batch_size, image_size/8, image_size/8, 256]
                net = slim.batch_norm(net, scope='bn3')
                net = slim.conv2d(net, 128, [init_size, init_size], padding='VALID', scope='conv4')
                net = slim.batch_norm(net, scope='bn4')
                fmap = slim.flatten(net)  # [batch_size, 128]
                logits = slim.fully_connected(fmap, hot_size, activation_fn=None, scope='logits')
                # [batch_size, hot_size]
                hot_code = tf.nn.softmax(logits, name='hot_code')
                # hot_code = tf.nn.sigmoid(logits)
                return mid, logits, hot_code

def Encoder_calm(mid, calm_size, reuse=False, is_training=True, name='Encoder_calm'):
    # mid_size = 8
    final_size = 4
    with tf.variable_scope(name, reuse=reuse):
        with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=None, stride=2,
                            weights_initializer=tf.contrib.layers.xavier_initializer()):
            with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                                activation_fn=tf.nn.relu, is_training=is_training):
                net = slim.conv2d(mid, 256, [1, 1], stride=1, scope='conv3')
                # [batch_size, mid_size, mid_size, 256]
                net = slim.batch_norm(net, scope='bn3')
                net = slim.conv2d(net, 128, [3, 3], scope='conv4')
                # [batch_size, mid_size/2, mid_size/2, 128]
                net = slim.batch_norm(net, scope='bn4')
                calm_code = slim.conv2d(net, calm_size, [final_size, final_size],
                                            padding='VALID', scope='conv5')
                # [batch_size, 1, 1, calm_size]
                calm_code = slim.flatten(calm_code)  # [batch_size, calm_size]
                return calm_code

def Decoder_hot(hot_code, image_size, reuse=False, is_training=True, name='Decoder_hot'):
    init_size = int(image_size/4)
    # hot_code = tf.log(hot_code)  # [batch_size, hot_size]
    hot_code = slim.fully_connected(hot_code, 128, activation_fn=None)  # [batch_size, 128]
    inputs = tf.expand_dims(hot_code, 1)  # [batch_size, 1, 128]
    inputs = tf.expand_dims(inputs, 1, name='inputs')  # [batch_size, 1, 1, 128]
    with tf.variable_scope(name, reuse=reuse):
        with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=None, stride=2,
                            weights_initializer=tf.contrib.layers.xavier_initializer()):
            with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                                activation_fn=tf.nn.relu, is_training=is_training):
                net = slim.conv2d_transpose(inputs, 64, [init_size, init_size], padding='VALID',
                                            scope='conv_t1')
                # [batch_size, image_size/4, image_size/4, 64]
                net = slim.batch_norm(net, scope='bn1')
                net = slim.conv2d_transpose(net, 32, [3, 3], padding='SAME', scope='conv_t2')
                # [batch_size, image_size/2, image_size/2, 32]
                net = slim.batch_norm(net, scope='bn2')
                return net

def Decoder_calm(calm_code, image_size, reuse=False, is_training=True, name='Decoder_calm'):
    mid_size = int(image_size/2)
    batch_size = 100
    inputs = slim.fully_connected(calm_code, mid_size*mid_size*16, activation_fn=None)
    # [batch_size, mid_size*mid_size*16]
    inputs = tf.reshape(inputs, [batch_size, mid_size, mid_size, 16])
    # [batch_size, mid_size, mid_size, 16]
    with tf.variable_scope(name, reuse=reuse):
        with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=None, stride=1,
                            weights_initializer=tf.contrib.layers.xavier_initializer()):
            with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                                activation_fn=tf.nn.relu, is_training=is_training):
                net = slim.conv2d_transpose(inputs, 32, [1, 1], scope='conv_t1')
                # [batch_size, mid_size, mid_size, 32]
                net = slim.batch_norm(net, scope='bn1')
                net = slim.conv2d_transpose(net, 32, [2, 2], scope='conv_t2')
                # [batch_size, mid_size, mid_size, 32]
                net = slim.batch_norm(net, scope='bn2')
                return net

def Decoder_image(mid_hot, mid_calm, reuse=False, is_training=True, name='Decoder_image'):
    inputs = tf.concat([mid_hot, mid_calm], axis=3)  # [batch_size, mid_size, mid_size, 64]
    with tf.variable_scope(name, reuse=reuse):
        with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=None,
                            weights_initializer=tf.contrib.layers.xavier_initializer()):
            with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                                activation_fn=tf.nn.relu, is_training=is_training):
                net = slim.conv2d_transpose(inputs, 32, [3, 3], stride=2, scope='conv_t3')
                # [batch_size, image_size, image_size, 32]
                net = slim.batch_norm(net, scope='bn3')
                net = slim.conv2d_transpose(net, 3, [1, 1], activation_fn=tf.tanh, stride=1,
                                            scope='conv_t4')
                # [batch_size, image_size, image_size, 3]
                return net

def Decoder_ctnn_shallow(latent_code, image_size, code_size, reuse=False, is_training=True,
                         name='Decoder_cnnt_shallow'):
    # latent_code: [batch_size, code_size]
    inputs = tf.expand_dims(latent_code, 1)
    inputs = tf.expand_dims(inputs, 1, name='inputs')  # [batch_size, 1, 1, code_size]
    init_size = int(image_size/8)
    with tf.variable_scope(name, reuse=reuse):
        with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=None, stride=2,
                            weights_initializer=tf.contrib.layers.xavier_initializer()):
            with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                                activation_fn=tf.nn.relu, is_training=is_training):
                net = slim.conv2d_transpose(inputs, 128, [init_size, init_size],
                                            padding='VALID', scope='conv_t1')
                # [batch_size, init_size, init_size, 128]
                net = slim.batch_norm(net, scope='bn1')
                net = slim.conv2d_transpose(net, 256, [3, 3], scope='conv_t2')
                # [batch_size, 2*init_size, 2*init_size, 256]
                net = slim.batch_norm(net, scope='bn2')
                net = slim.conv2d_transpose(net, 64, [3, 3], scope='conv_t3')
                # [batch_size, 4*init_szie, 4*init_size, 64]
                net = slim.batch_norm(net, scope='bn3')
                net = slim.conv2d_transpose(net, 3, [3, 3], activation_fn=tf.nn.tanh,
                                            scope='conv_t4')
                # [batch_size, image_size, image_size, 3]
                return net

def Discriminator_lc_fn(latent_code, code_size, reuse=False, name='Discriminator_lc_fn'):
    # latent_code: [batch_size, code_size]
    with tf.variable_scope(name, reuse=reuse):
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
            net = slim.fully_connected(latent_code, 256, scope='fn1')
            net = slim.fully_connected(net, 64, scope='fn2')
            net = slim.fully_connected(net, 16, scope='fn3')
            net = slim.fully_connected(net, 1, activation_fn=None, scope='logits')
            return net

def Discriminator_image_cnn(images, image_size, reuse=False, is_training=True,
                            name='Discriminator_image_cnn'):
    # images: [batch_size, image_size, image_size, 3]
    final_size = int(image_size / 8)
    with tf.variable_scope(name, reuse=reuse):
        with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=None, stride=2,
                            weights_initializer=tf.contrib.layers.xavier_initializer()):
            with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                                activation_fn=tf.nn.relu, is_training=is_training):
                net = slim.conv2d(images, 64, [3, 3], scope='conv1')
                # [batch_size, image_size/2, image_size/2, 64]
                net = slim.batch_norm(net, scope='bn1')
                net = slim.conv2d(net, 128, [3, 3], scope='conv2')
                # [batch_size, image_size/4, image_size/4, 128]
                net = slim.batch_norm(net, scope='bn2')
                net = slim.conv2d(net, 64, [3, 3], scope='conv3')
                # [batch_size, image_size/8, image_size/8, 64]
                net = slim.batch_norm(net, scope='bn3')
                net = slim.conv2d(net, 1, [final_size, final_size], padding='VALID', scope='conv4')
                # [batch_size, 1, 1, 1]
                net = slim.flatten(net, scope='logits')
                # [batch_size, 1]
                return net