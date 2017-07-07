import tensorflow as tf
import numpy as np
from network import Network
from VGGnet_test import VGGnet_test

# name wrapper
def nw(name, prefix='forked_'):
    #return prefix + name
    return name

class forked_VGGnet(VGGnet_test):
    def __init__(self, train=True, conv1_5_trainable=[False, False, False, False, False]):
        assert len(tf.all_variables()) == 0
        # init original VGG test net
        VGGnet_test.__init__(self, False, conv1_5_trainable)
        # Note: We don't consider the background class at the forked_VGGnet!!!!!!!
        self.n_classes -= 1

        # Use a placeholder as the classification targets.
        # There are 4 placeholders: data, im_info, keep_prob and classes 
        self.classes = tf.placeholder(tf.float32, shape=[None, self.n_classes])
        
        # Construct the hack roi, whose receptive field equals to the entire image.
        # im_info: a list of [image_height, image_width, scale_ratios]
        self.layers['hack_roi'] = tf.py_func(
            lambda im_info: np.array([[0, 0, 0, im_info[0][1], im_info[0][0]]], dtype=np.float32), # [[image_index, x0,y0,x1,y1]] 
            [self.im_info], 
            tf.float32, name='hack_roi')

        # Construct the forked classification branch.
        if train:   # for training phase
            (self.feed('conv5_3', 'hack_roi')
                .roi_pool(7, 7, 1.0 / 16, name=nw('pool5'))
                .fc(4096, name=nw('fc6'), trainable=True)
                .dropout(0.5, name=nw('drop6'))
                .fc(4096, name=nw('fc7'), trainable=True)
                .dropout(0.5, name=nw('drop7'))
                .fc(self.n_classes, activation=None, name=nw('cls_scores'), trainable=True))
        else:   # for test phase, just remove the dropout layer.
            (self.feed('conv5_3', 'hack_roi')
                .roi_pool(7, 7, 1.0 / 16, name=nw('pool5'))
                .fc(4096, name=nw('fc6'), trainable=False)
                .fc(4096, name=nw('fc7'), trainable=False)
                .fc(self.n_classes, activation=None, name=nw('cls_scores'), trainable=False))

        # Classification loss.
        self.outputs = self.layers[nw('cls_scores')]
        self.cls_loss = tf.losses.sigmoid_cross_entropy(self.classes, self.outputs)
        self.outputs = tf.nn.sigmoid(self.outputs)

        # Variable collector.
        self.det_variables = [var for var in tf.all_variables() if not var.name.startswith('cls_scores')]
        #self.cls_variables = [var for var in tf.all_variables() if var.name.startswith('forked')]
        self.trainable_variables = tf.trainable_variables()

        for layer in ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'pool5', 'fc6', 'fc7', 'cls_scores']:
            h = self.get_output(layer)
            tf.summary.histogram('Activation/'+h.name, h)
            tf.summary.scalar('Sparsity/'+h.name, tf.nn.zero_fraction(h))

def visual():
    fvgg = forked_VGGnet(True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("tmp/", sess.graph)
    writer.flush()

