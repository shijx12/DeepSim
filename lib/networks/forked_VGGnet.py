import tensorflow as tf
import numpy as np
from network import Network
from VGGnet_test import VGGnet_test

# name wrapper
def nw(name, prefix='forked_'):
    return prefix + name

class forked_VGGnet(VGGnet_test):
    def __init__(self, trainable=True):
        assert len(tf.all_variables()) == 0
        # init original VGG test net
        VGGnet_test.__init__(self, False)
        # Note: We don't consider the background class at the forked_VGGnet!!!!!!!
        self.n_classes -= 1

        # Use a placeholder as the classification targets.
        # There are 4 placeholders: data, im_info, keep_prob and classes 
        self.classes = tf.placeholder(tf.float32, shape=[None, self.n_classes])
        
        self.det_variables = tf.all_variables()
        # Construct the hack roi, whose receptive field equals to the entire image.
        self.layers['hack_roi'] = tf.py_func(
            lambda im_info: np.array([[0, 0, 0, im_info[0][0], im_info[0][1]]], dtype=np.float32), 
            [self.im_info], 
            tf.float32, name='hack_roi')

        # Construct the forked classification branch.
        (self.feed('conv5_3', 'hack_roi')
                .roi_pool(7, 7, 1.0 / 16, name=nw('pool5'))
                .fc(4096, name=nw('fc6'), trainable=trainable)
                .dropout(0.5, name=nw('drop6'))
                .fc(4096, name=nw('fc7'), trainable=trainable)
                .dropout(0.5, name=nw('drop7'))
                .fc(self.n_classes, relu=False, name=nw('cls_scores'), trainable=trainable))

        # Classification loss.
        self.outputs = self.layers[nw('cls_scores')]
        self.cls_loss = tf.losses.sigmoid_cross_entropy(self.classes, self.outputs)

        # Variable collector.
        self.cls_variables = [var for var in tf.all_variables() if var not in self.det_variables]
        self.trainable_variables = tf.trainable_variables()

def visual():
    fvgg = forked_VGGnet(True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("tmp/", sess.graph)
    writer.flush()

