from lib.datasets.factory import get_imdb
import numpy as np
import tensorflow as tf
import cv2
import random
import cfg

class DataFetcher:
    def __init__(self, imdb_name, resize=True):
        imdb = get_imdb(imdb_name)
        # Ignore the background class!!! So ['gt_classes'] must minus 1.
        self.classes = [ np.zeros(imdb.num_classes - 1) for i in range(imdb.num_images) ]
        for i, anno in enumerate(imdb.gt_roidb()):
            np.put(self.classes[i], map(lambda x: x-1, anno['gt_classes']), 1)
            # np.put(self.classes[i], random.choice(map(lambda x: x-1, anno['gt_classes'])), 1)
        self.images = [ imdb.image_path_at(i) for i in range(imdb.num_images) ]
        assert len(self.classes) == len(self.images)

        self._perm = np.random.permutation(np.arange(len(self.images)))
        self._cur = 0
        self.resize = resize

    def nextbatch(self, batch_size=1):
        # if all images have been trained, permuate again.
        blobs = {'data':[], 'path':[], 'classes':[], 'keep_prob':[], 'im_info':[]}
        for batch in range(batch_size):
            if self._cur >= len(self.images):
                self._cur = 0
                self._perm = np.random.permutation(np.arange(len(self.images)))
            i = self._perm[self._cur]
            self._cur += 1
            im = cv2.imread(self.images[i]).astype(np.float32, copy=False)
            if self.resize:
                im = np.stack([cv2.resize(im[:,:,i], (cfg.RESIZED_SIZE, cfg.RESIZED_SIZE)) for i in range(im.shape[2])], axis=2)
            blobs['data'].append(im) 
            blobs['path'].append(self.images[i])
            blobs['classes'].append(self.classes[i])
            blobs['keep_prob'].append(0.5)
            # im_info: a list of [image_height, image_width, scale_ratios]
            # im_scale=1, that is, we don't scale the original image size.
            blobs['im_info'].append([im.shape[0], im.shape[1], 1])
        return blobs

def crop(image, resized_size, cropped_size):
    # image is of arbitrary size.
    # return a Tensor representing image of size cropped_size x cropped_size
    image = tf.image.resize_images(image, [resized_size, resized_size], method=tf.image.ResizeMethod.AREA)
    offset = tf.cast(tf.floor(tf.random_uniform([2], 0, resized_size - cropped_size + 1)), dtype=tf.int32)
    image = tf.image.crop_to_bounding_box(image, offset[0], offset[1], cropped_size, cropped_size)
    return image

def subtract_mean(image):
    # image is a Tensor.
    # return a Tensor.
    image = tf.cast(image, dtype=tf.float32)
    return image - tf.convert_to_tensor(cfg.PIXEL_MEANS, dtype=tf.float32)

def prep(image):
    # change range from [0, 256) to [-1, 1]
    # image is a Tensor.
    # return a float32 Tensor.
    image = tf.cast(image, dtype=tf.float32)
    return (image / 255.0) * 2 - 1


def invprep(image):
    # change range from [-1, 1] to [0, 256)
    # image is a float32 Tensor.
    # return a uint8 Tensor.
    image = (image + 1) / 2.0 * 255.9
    return image

def bgr2rgb(image):
    image = tf.cast(image, dtype=tf.uint8)
    return image[:,:,:,::-1]


def make_var(name, shape, initializer=None, trainable=True, regularizer=None):
    return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)

def l2_regularizer(weight_decay=0.0005, scope=None):
    def regularizer(tensor):
        with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
            l2_weight = tf.convert_to_tensor(weight_decay, dtype=tensor.dtype.base_dtype, name='weight_decay')
            return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
    return regularizer

def batch_norm(input, scope='batchnorm'):
    with tf.variable_scope(scope):
        input = tf.identity(input)
        dims = input.get_shape()
        if len(dims) == 4:
            channels = dims[3]
            offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
            scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
            mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        elif len(dims) == 2:
            channels = dims[1]
            offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
            scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
            mean, variance = tf.nn.moments(input, axes=[0], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized

def leaky_relu(input, alpha=0.3, name='leaky_relu'):
    return tf.maximum(alpha*input, input, name)

def conv(input, k_h, k_w, c_o, s_h, s_w, name, biased=True, activation='relu', bn=False, init='msra', pad='SAME', trainable=True):
    c_i = input.get_shape()[-1] # channel_input
    with tf.variable_scope(name) as scope:
        if init == 'msra':
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
        else:
            raise Exception('Invalid init')
        kernel = make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable, regularizer=l2_regularizer(cfg.WEIGHT_DECAY))
        h = tf.nn.conv2d(input, kernel, [1, s_h, s_w, 1], padding=pad)
        if biased:
            init_bias = tf.constant_initializer(0.0)
            bias = make_var('biases', [c_o], init_bias, trainable)
            h = tf.nn.bias_add(h, bias)
        if bn:
            h = batch_norm(h)
        if activation == 'relu':
            h = tf.nn.relu(h)
        elif activation == 'leaky_relu':
            h = leaky_relu(h)
        return h

def upconv(input, c_o, ksize, stride, name, biased=False, activation='relu', bn=False, init='msra', pad='SAME', trainable=True):
    c_i = input.get_shape()[-1] # channel_input
    in_shape = tf.shape(input)
    if pad == 'SAME':
        output_shape = [in_shape[0], in_shape[1]*stride, in_shape[2]*stride, c_o]
    else:
        raise Exception('Sorry not support padding VALID')
    kernel_shape = [ksize, ksize, c_o, c_i]
    with tf.variable_scope(name) as scope:
        if init == 'msra':
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
        else:
            raise Exception('Invalid init')
        kernel = make_var('weights', kernel_shape, init_weights, trainable, regularizer=l2_regularizer(cfg.WEIGHT_DECAY))
        h = tf.nn.conv2d_transpose(input, kernel, output_shape, [1, stride, stride, 1], padding=pad)
        h = tf.reshape(h, output_shape) # reshape is necessary
        if biased:
            init_bias = tf.constant_initializer(0.0)
            bias = make_var('biases', [c_o], init_bias, trainable)
            h = tf.nn.bias_add(h, bias)
        if bn:
            h = batch_norm(h)
        if activation == 'relu':
            h = tf.nn.relu(h)
        elif activation == 'leaky_relu':
            h = leaky_relu(h)
        return h

def max_pool(input, k_h, k_w, s_h, s_w, name, pad='SAME'):
    return tf.nn.max_pool(input, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=pad, name=name)

def avg_pool(input, k_h, k_w, s_h, s_w, name, pad='SAME'):
    return tf.nn.avg_pool(input, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=pad, name=name)

def fc(input, c_o, name, biased=True, activation='relu', bn=False, init='msra', trainable=True):
    c_i = input.get_shape()[-1]
    with tf.variable_scope(name) as scope:
        if init == 'msra':
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
        else:
            raise Exception('Invalid init')
        weights = make_var('weights', [c_i, c_o], init_weights, trainable, regularizer=l2_regularizer(cfg.WEIGHT_DECAY))
        h = tf.matmul(input, weights)
        if biased:
            init_bias = tf.constant_initializer(0.0)
            bias = make_var('biases', [c_o], init_bias, trainable)
            h = tf.nn.bias_add(h, bias)
        if bn:
            h = batch_norm(h)
        if activation == 'relu':
            h = tf.nn.relu(h)
        elif activation == 'leaky_relu':
            h = leaky_relu(h)
        return h


def sum_act(h, sparsity=False):
    tf.summary.histogram('activation/'+h.name, h)
    if sparsity:
        tf.summary.scalar('sparsity/'+h.name, tf.nn.zero_fraction(h))
