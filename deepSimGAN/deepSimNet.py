import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__)))) # To find lib and util
import tensorflow as tf
import numpy as np
from lib.networks.network import Network
import util
import cfg


class deepSimNet(Network):
    def __init__(self, recon_weight, feat_weight, dis_weight, gan_type, trainable=True):
        self.height = cfg.HEIGHT
        self.width = cfg.WIDTH
        assert self.height == self.width
        self.n_classes = cfg.N_CLASSES
        self.original_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='original_image')
        self.real_image = util.crop(self.original_image, cfg.RESIZED_SIZE, cfg.HEIGHT)  # crop to fix size
        self.real_image = util.prep(self.real_image)    # range [-1, 1]
        self.layers = dict()
        self.trainable = trainable
        self.recon_weight, self.feat_weight, self.dis_weight = recon_weight, feat_weight, dis_weight
        self.gan_type = gan_type
        self.setup(False)

    def setup(self, debug=True):
        with tf.name_scope('real_encoder'):
            with tf.variable_scope(''): # NOTE: using an empty string as variable_scope, aims to be consistent with forked_VGGnet's variable name and to restore from trained forked_VGGnet's model
                self.inv_h, self.real_cmp_h, _ = self.Encoder(self.real_image)

        with tf.name_scope('generator'):
            with tf.variable_scope('generator'):
                self.fake_image = self.Generator(self.inv_h)

        with tf.name_scope('fake_encoder'):
            with tf.variable_scope('', reuse=True):
                _, self.fake_cmp_h, __ = self.Encoder(self.fake_image)

        with tf.name_scope('real_disc'):
            with tf.variable_scope('discriminator'):
                self.real_score_logit = self.Discriminator(self.real_image, self.real_cmp_h)
        with tf.name_scope('fake_disc'):
            with tf.variable_scope('discriminator', reuse=True):
                self.fake_score_logit = self.Discriminator(self.fake_image, self.fake_cmp_h)

        # generator trainable variables
        self.gen_variables = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
        # discriminator trainable variables
        self.dis_variables = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
        # encoder variables to restore
        self.enc_variables = [var for var in tf.all_variables() if var not in self.gen_variables and var not in self.dis_variables]

        # losses
        self.recon_loss = self.recon_weight * tf.reduce_mean(tf.square(self.real_image - self.fake_image))
        self.feat_loss = self.feat_weight * tf.reduce_mean(tf.square(self.real_cmp_h - self.fake_cmp_h))
        if self.gan_type == 'wgan':
            self.gen_dis_loss = self.dis_weight * tf.reduce_mean(self.fake_score_logit)
            self.dis_loss = self.dis_weight * tf.reduce_mean(self.real_score_logit - self.fake_score_logit)
        elif self.gan_type == 'lsgan':
            self.gen_dis_loss = self.dis_weight * tf.reduce_mean(tf.square(self.fake_score_logit - 1))
            self.dis_loss = self.dis_weight * tf.reduce_mean(tf.square(self.real_score_logit - 1) + tf.square(self.fake_score_logit))
        self.gen_loss = self.recon_loss + self.feat_loss + self.gen_dis_loss 

        if debug:
            print ' ----- DEBUG NETWORK KEY TENSOR ----'
            print self.inv_h
            print self.real_cmp_h, self.fake_cmp_h
            print self.fake_image
            print self.real_score_logit, self.fake_score_logit
            print 'gen_variables', self.gen_variables
            print 'dis_variables', self.dis_variables
            print 'enc_variables', self.enc_variables
            print ' -----------------------------------'


    def Encoder(self, image, inv_layer='fc6', cmp_layer='pool5', score_layer='cls_score'): # Untrainable at all. Must be restored !!
        image.set_shape([None, self.height, self.width, 3])
        image = util.subtract_mean(util.invprep(image)) # change range from [-1,1] to [0, 256] and then subtract mean pixel
        self.layers['image'] = image
        (self.feed('image')
             .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
             .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
             .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1', trainable=False)
             .conv(3, 3, 256, 1, 1, name='conv3_2', trainable=False)
             .conv(3, 3, 256, 1, 1, name='conv3_3', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1', trainable=False)
             .conv(3, 3, 512, 1, 1, name='conv4_2', trainable=False)
             .conv(3, 3, 512, 1, 1, name='conv4_3', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1', trainable = False)
             .conv(3, 3, 512, 1, 1, name='conv5_2', trainable = False)
             .conv(3, 3, 512, 1, 1, name='conv5_3', trainable = False)) # 14x14x512
        #self.layers['hack_roi'] = tf.constant(np.array([[0, 0, 0, self.width, self.height]], dtype=np.float32), name='hack_roi') # [[image_index, x0,y0,x1,y1]] 
        #(self.feed('conv5_3', 'hack_roi')
        #     .roi_pool(7, 7, 1.0 / 16, name='forked_pool5') # 7x7x512. NOTE: roi_pool limits that batch size must be 1. 
        (self.feed('conv5_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool5') # 7x7x512. Totally equivalent to the commented hack_roi.
             .fc(4096, name='fc6', trainable=False)
             .fc(4096, name='fc7', trainable=False)
             .fc(self.n_classes, activation=None, name='cls_score', trainable=False))
        # return inverted_h, compared_h, cls_scores
        return self.get_output(inv_layer), self.get_output(cmp_layer), self.get_output(score_layer)

    def Generator(self, inv_h):
        self.layers = {}
        self.layers['inv_h'] = inv_h
        act = 'relu'
        bn = True
        (self.feed('inv_h')
             .fc(4096, name='defc5', activation=act, init='msra', bn=True, trainable=self.trainable)
             .h_reshape(shape=[4, 4, 256], name='reshape_defc5')
             .upconv(shape=None, c_o=256, ksize=4, stride=2, name='deconv5', activation=act, init='msra', bn=bn, trainable=self.trainable)   # 8x8, 
             .upconv(shape=None, c_o=512, ksize=3, stride=1, name='deconv5_1', activation=act, init='msra', bn=bn, trainable=self.trainable) # 8x8
             .upconv(shape=None, c_o=256, ksize=4, stride=2, name='deconv4', activation=act, init='msra', bn=bn, trainable=self.trainable)   # 16x16
             .upconv(shape=None, c_o=256, ksize=3, stride=1, name='deconv4_1', activation=act, init='msra', bn=bn, trainable=self.trainable) # 16x16
             .upconv(shape=None, c_o=128, ksize=4, stride=2, name='deconv3', activation=act, init='msra', bn=bn, trainable=self.trainable)   # 32x32
             .upconv(shape=None, c_o=128, ksize=3, stride=1, name='deconv3_1', activation=act, init='msra', bn=bn, trainable=self.trainable) # 32x32
             .upconv(shape=None, c_o=64,  ksize=4, stride=2, name='deconv2', activation=act, init='msra', bn=bn, trainable=self.trainable)   # 64x64
             .upconv(shape=None, c_o=32,  ksize=4, stride=2, name='deconv1', activation=act, init='msra', bn=bn, trainable=self.trainable)   # 128x128
             .upconv(shape=None, c_o=3,   ksize=4, stride=2, activation=None, name='deconv0', init='msra', bn=False, trainable=self.trainable))   # 256x256

        for k, var in self.layers.items():
            tf.summary.histogram('activation/'+var.name, var)
        h_offset = (256 - self.height) / 2
        w_offset = (256 - self.width) / 2
        fake_image = tf.image.crop_to_bounding_box(self.get_output('deconv0'), h_offset, w_offset, self.height, self.width)
        fake_image = tf.nn.tanh(fake_image, name='tanh')    # range (-1, 1)
        tf.summary.histogram('activation/'+fake_image.name, fake_image)
        return fake_image
             
       
    def Discriminator(self, image, feature):
        image.set_shape([None, self.height, self.width, 3])
        self.layers['image'] = image
        self.layers['feat'] = feature
        act = 'leaky_relu'
        bn = True
        (self.feed('image')
             .conv(7, 7, 32, 4, 4, name='conv1', padding='VALID',activation=act,init='msra',bn=False,trainable=self.trainable) # 56x56
             .conv(5, 5, 64, 1, 1, name='conv2', padding='VALID',activation=act,init='msra',bn=bn,trainable=self.trainable) # 52x52
             .conv(3, 3, 128, 2,2, name='conv3', padding='VALID',activation=act,init='msra',bn=bn,trainable=self.trainable) # 25x25
             .conv(3, 3, 256, 1,1, name='conv4', padding='VALID',activation=act,init='msra',bn=bn,trainable=self.trainable) # 23x23
             .conv(3, 3, 256, 2,2, name='conv5', padding='VALID',activation=act,init='msra',bn=bn,trainable=self.trainable) # 11x11
             .avg_pool(11, 11, 11, 11, name='pool5', padding='VALID')
             .reshape_toh(name='pool5_reshape')) # 256
        (self.feed('feat')
             .fc(1024, name='feat_fc1', activation=act, init='msra', bn=bn, trainable=self.trainable)
             .fc(512, name='feat_fc2', activation=act, init='msra', bn=bn, trainable=self.trainable))
        (self.feed('pool5_reshape', 'feat_fc2')
             .concat(axis=1, name='concat_fc5')) # 256+512=768
        if self.trainable:  # train phase
            (self.feed('concat_fc5')
                 .dropout(0.5, name='drop5')
                 .fc(512, name='fc6', activation=act, init='msra', bn=bn, trainable=self.trainable)
                 .dropout(0.5, name='drop6')
                 .fc(1, name='fc7', activation=None, init='msra', bn=False, trainable=self.trainable))
        else:               # test phase
            (self.feed('concat_fc5')
                 .fc(512, name='fc6', activation=act, init='msra', bn=bn, trainable=self.trainable)
                 .fc(1, name='fc7', activation=None, init='msra', bn=False, trainable=self.trainable))
        return self.get_output('fc7')       # range (-inf, inf)


def test():
    net = deepSimNet()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./tmp/', sess.graph)
        writer.flush()


if __name__ == '__main__':
    test()
