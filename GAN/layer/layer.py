'''
covlution layer，initialization。。。。
'''
import tensorflow as tf
import scipy.misc
import numpy as np


# Weight initialization (Xavier's init)
def weight_xavier_init(shape, n_inputs, n_outputs, variable_name, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        initial = tf.random_uniform(shape, -init_range, init_range)
        return tf.get_variable(name=variable_name, initializer=initial, trainable=True)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.get_variable(name=variable_name, initializer=initial, trainable=True)


# Bias initialization
def bias_variable(shape, variable_name):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(initializer=initial, trainable=True, name=variable_name)


# 2D convolution
def convolution_2d(x, w, stride=1):
    return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')


# 2D deconvolution
def deconvolution_2d(x, W, strides=2):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1] * strides, x_shape[2] * strides, x_shape[3] // strides])
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, strides, strides, 1], padding='SAME')


# Max Pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Average Pooling
def average_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# resnet add_connect
def add_connect(x1, x2):
    if x1.get_shape().as_list()[3] != x2.get_shape().as_list()[3]:
        residual_connection = x2 + tf.pad(x1, [[0, 0], [0, 0], [0, 0],
                                               [0, x2.get_shape().as_list()[3] - x1.get_shape().as_list()[3]]])
    else:
        residual_connection = x1 + x2
    return residual_connection


# calculate several tensors mean
def AverageTensors(inputs):
    output = inputs[0]
    for i in range(1, len(inputs)):
        output += inputs[i]
    return output / len(inputs)


# if x>0,using x,if x<0,using leak*x
def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


# Unet crop and concat
def crop_and_concat(x1, x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)


# Modularity Net
def conv_relu(x, kernel_shape, scope=None):
    if len(kernel_shape) != 4:
        print("kernel_shape is not right")
    with tf.name_scope(scope):
        g_w = weight_xavier_init(shape=kernel_shape, n_inputs=kernel_shape[-2], n_outputs=kernel_shape[-1],
                                 variable_name=scope + 'g_w')
        g_b = bias_variable(shape=[kernel_shape[-1]], variable_name=scope + 'g_b')
        g = convolution_2d(x, g_w, 2)
        g = g + g_b
        g = tf.nn.relu(g)
        return g


def conv_leaky_relu(x, kernel_shape, scope=None):
    if len(kernel_shape) != 4:
        print("kernel_shape is not right")
    with tf.name_scope(scope):
        d_w = weight_xavier_init(shape=kernel_shape, n_inputs=kernel_shape[-2], n_outputs=kernel_shape[-1],
                                 variable_name=scope + 'd_w')
        d_b = bias_variable(shape=[kernel_shape[-1]], variable_name=scope + 'd_b')
        d = convolution_2d(x, d_w, stride=2)
        d = d + d_b
        d = leaky_relu(d)
        return d


def full_conv(x, kernel_shape, active=True, scope=None):
    if len(kernel_shape) != 2:
        print("kernel_shape is not right")
    with tf.name_scope(scope):
        d_w = weight_xavier_init(shape=kernel_shape, n_inputs=kernel_shape[-2], n_outputs=kernel_shape[-1],
                                 variable_name=scope + 'd_w')
        d_b = bias_variable(shape=[kernel_shape[-1]], variable_name=scope + 'd_b')
        d = tf.matmul(x, d_w)
        d = d + d_b
        if active == True:
            d = leaky_relu(d)
        return d


def conv_sigmod(x, kernel_shape, scope=None):
    if len(kernel_shape) != 4:
        print("kernel_shape is not right")
    with tf.name_scope(scope):
        g_w = weight_xavier_init(shape=kernel_shape, n_inputs=kernel_shape[-2], n_outputs=kernel_shape[-1],
                                 variable_name=scope + 'g_w')
        g_b = bias_variable(shape=[kernel_shape[-1]], variable_name=scope + 'g_b')
        g = convolution_2d(x, g_w)
        g = g + g_b
        g = tf.nn.sigmoid(g)
        return g


def deconv_relu(x, kernel_shape, scope=None):
    if len(kernel_shape) != 4:
        print("kernel_shape is not right")
    with tf.name_scope(scope):
        g_w = weight_xavier_init(shape=kernel_shape, n_inputs=kernel_shape[-1], n_outputs=kernel_shape[-2],
                                 variable_name=scope + 'g_w')
        g_b = bias_variable(shape=[kernel_shape[-2]], variable_name=scope + 'g_b')
        g = deconvolution_2d(x, g_w)
        g = g + g_b
        g = tf.nn.relu(g)
        return g


def save_images(images, size, path):
    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]
    merge_img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j * h:j * h + h, i * w:i * w + w, :] = image
    return scipy.misc.imsave(path, merge_img)
