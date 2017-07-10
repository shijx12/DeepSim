from lib.datasets.factory import get_imdb
import numpy as np
import tensorflow as tf
import cv2
import random
import cfg

class DataFetcher:
    def __init__(self, imdb_name):
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
            blobs['data'].append(im) 
            blobs['path'].append(self.images[i])
            blobs['classes'].append(self.classes[i])
            blobs['keep_prob'].append(0.5)
            # im_info: a list of [image_height, image_width, scale_ratios]
            # im_scale=1, that is, we don't scale the original image size.
            blobs['im_info'].append([im.shape[1], im.shape[2], 1])
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
