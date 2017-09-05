import tensorflow as tf
import numpy as np
from util import *
import cfg


class deepSimNet():
    def __init__(self, batch_size, recon_weight, feat_weight, dis_weight, gan_type, trainable=True):
        self.original_image = tf.placeholder(tf.float32, shape=[batch_size, None, None, 3], name='original_image')
        self.real_image = crop(self.original_image, cfg.RESIZED_SIZE, cfg.IMAGE_SIZE)  # crop to fixed size
        self.real_image = prep(self.real_image)    # range [-1, 1]
        debug = False
        self.noise_sigma = tf.placeholder(tf.float32, shape=(), name='noise_sigma')

        with tf.name_scope('real_encoder'):
            with tf.variable_scope(''): # NOTE: using an empty string as variable_scope, aims to be consistent with EncoderNet's variable name and to restore from trained EncoderNet's model
                self.inverted_h, self.real_cmp_h, _ = Encoder(self.real_image)

        with tf.name_scope('generator'):
            with tf.variable_scope('generator'):
                self.fake_image = Generator(self.inverted_h, trainable)

        with tf.name_scope('fake_encoder'):
            with tf.variable_scope('', reuse=True):
                _, self.fake_cmp_h, __ = Encoder(self.fake_image)

        with tf.name_scope('real_disc'):
            with tf.variable_scope('discriminator'):
                self.real_score_logit = Discriminator(self.real_image, self.real_cmp_h, trainable, self.noise_sigma)
        with tf.name_scope('fake_disc'):
            with tf.variable_scope('discriminator', reuse=True):
                self.fake_score_logit = Discriminator(self.fake_image, self.fake_cmp_h, trainable, self.noise_sigma)

        # generator trainable variables
        self.gen_variables = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
        # discriminator trainable variables
        self.dis_variables = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
        # encoder variables to restore
        self.enc_variables = [var for var in tf.all_variables() if var not in self.gen_variables and var not in self.dis_variables]

        # losses
        self.recon_loss = tf.reduce_mean(tf.square(self.real_image - self.fake_image))
        self.feat_loss = tf.reduce_mean(tf.square(self.real_cmp_h - self.fake_cmp_h))
        if gan_type == 'wgan':
            self.gen_dis_loss = tf.reduce_mean(self.fake_score_logit)
            self.dis_loss = tf.reduce_mean(self.real_score_logit - self.fake_score_logit)
        elif gan_type == 'lsgan':
            self.gen_dis_loss = tf.reduce_mean(tf.square(self.fake_score_logit - 1))
            self.dis_loss = tf.reduce_mean(tf.square(self.real_score_logit - 1) + tf.square(self.fake_score_logit))
        elif gan_type == 'gan':
            g_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.fake_score_logit,
                    labels=tf.ones_like(self.fake_score_logit)
                    )
            self.gen_dis_loss = tf.reduce_mean(g_fake_loss) 
            d_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.real_score_logit,
                    labels=tf.ones_like(self.real_score_logit)
                    )
            d_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.fake_score_logit,
                    labels=tf.ones_like(self.fake_score_logit)
                    )
            self.dis_loss = tf.reduce_mean(d_real_loss + d_fake_loss)
        self.gen_loss = recon_weight * self.recon_loss + feat_weight * self.feat_loss + dis_weight * self.gen_dis_loss 

        if debug:
            print ' ----- DEBUG NETWORK KEY TENSOR ----'
            print self.inverted_h
            print self.real_cmp_h, self.fake_cmp_h
            print self.fake_image
            print self.real_score_logit, self.fake_score_logit
            print 'gen_variables', self.gen_variables
            print 'dis_variables', self.dis_variables
            print 'enc_variables', self.enc_variables
            print ' -----------------------------------'

#######
# Untrainable at all. Must be restored !! 
# Here its structure is the same as EncoderNet.Encoder_net
# So it must be restored from EncoderNet's checkpoint 
######
def Encoder(image): 
    image.set_shape([None, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3])
    image = subtract_mean(invprep(image)) # change range from [-1,1] to [0, 256] and then subtract mean pixel
    h = conv(image, 3, 3, 64, 1, 1, name='conv1_1', trainable=False)
    h = conv(h, 3, 3, 64, 1, 1, name='conv1_2', trainable=False)
    h = max_pool(h, 2, 2, 2, 2, pad='VALID', name='pool1')
    h = conv(h, 3, 3, 128, 1, 1, name='conv2_1', trainable=False)
    h = conv(h, 3, 3, 128, 1, 1, name='conv2_2', trainable=False)
    h = max_pool(h, 2, 2, 2, 2, pad='VALID', name='pool2')
    h = conv(h, 3, 3, 256, 1, 1, name='conv3_1', trainable=False)
    h = conv(h, 3, 3, 256, 1, 1, name='conv3_2', trainable=False)
    h = conv(h, 3, 3, 256, 1, 1, name='conv3_3', trainable=False)
    h = max_pool(h, 2, 2, 2, 2, pad='VALID', name='pool3')
    h = conv(h, 3, 3, 512, 1, 1, name='conv4_1', trainable=False)
    h = conv(h, 3, 3, 512, 1, 1, name='conv4_2', trainable=False)
    h = conv(h, 3, 3, 512, 1, 1, name='conv4_3', trainable=False)
    h = max_pool(h, 2, 2, 2, 2, pad='VALID', name='pool4')
    h = conv(h, 3, 3, 512, 1, 1, name='conv5_1', trainable = False)
    h = conv(h, 3, 3, 512, 1, 1, name='conv5_2', trainable = False)
    h = conv(h, 3, 3, 512, 1, 1, name='conv5_3', trainable = False) # 14x14x512
    h = max_pool(h, 2, 2, 2, 2, pad='VALID', name='pool5') # 7x7x512
    h = tf.reshape(h, [-1, 7*7*512], name='reshape_pool5')
    compared_h = h  ##### compared_h: pool5
    h = fc(h, 4096, name='fc6', trainable=False)
    inverted_h = h  ##### inverted_h: fc6
    h = fc(h, 4096, name='fc7', trainable=False)
    h = fc(h, cfg.N_CLASSES, activation='', name='cls_score', trainable=False)
    # return inverted_h, compared_h, cls_score
    return inverted_h, compared_h, h


def Generator(inverted_h, trainable): 
    act = 'relu'
    bn = True
    h = fc(inverted_h, 4096, name='defc7', activation=act, init='msra', bn=True, trainable=trainable)
    h = fc(h, 4096, name='defc6', activation=act, init='msra', bn=True, trainable=trainable)
    h = fc(h, 4096, name='defc5', activation=act, init='msra', bn=True, trainable=trainable)
    h = tf.reshape(h, [-1, 4, 4, 256], name='reshape_defc5')
    h = upconv(h, c_o=256, ksize=4, stride=2, name='deconv5', activation=act, bn=bn, trainable=trainable)   # 8x8, 
    h = upconv(h, c_o=512, ksize=3, stride=1, name='deconv5_1', activation=act, bn=bn, trainable=trainable) # 8x8
    h = upconv(h, c_o=256, ksize=4, stride=2, name='deconv4', activation=act, bn=bn, trainable=trainable)   # 16x16
    h = upconv(h, c_o=256, ksize=3, stride=1, name='deconv4_1', activation=act, bn=bn, trainable=trainable) # 16x16
    h = upconv(h, c_o=128, ksize=4, stride=2, name='deconv3', activation=act, bn=bn, trainable=trainable)   # 32x32
    h = upconv(h, c_o=128, ksize=3, stride=1, name='deconv3_1', activation=act, bn=bn, trainable=trainable) # 32x32
    h = upconv(h, c_o=64,  ksize=4, stride=2, name='deconv2', activation=act, bn=bn, trainable=trainable)   # 64x64
    h = upconv(h, c_o=32,  ksize=4, stride=2, name='deconv1', activation=act, bn=bn, trainable=trainable)   # 128x128
    h = upconv(h, c_o=3,   ksize=4, stride=2, name='deconv0', activation='', bn=False, trainable=trainable)   # 256x256
    #tf.summary.histogram('activation/'+h.name, h)

    h_offset = (256 - cfg.IMAGE_SIZE) / 2
    w_offset = (256 - cfg.IMAGE_SIZE) / 2
    fake_image = tf.image.crop_to_bounding_box(h, h_offset, w_offset, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
    fake_image = tf.nn.tanh(fake_image, name='tanh')    # range (-1, 1)
    tf.summary.histogram('activation/'+fake_image.name, fake_image)
    return fake_image
             
       
def Discriminator(image, feature, trainable, noise_sigma):
    image.set_shape([None, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3])
    # add random noise to the input of discriminator. To make the traning of D harder.
    image = image + tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=noise_sigma)
    feature = feature + tf.random_normal(shape=tf.shape(feature), mean=0.0, stddev=2*noise_sigma)
    act = 'leaky_relu'
    bn = True
    pad = 'VALID'
    h = conv(image, 7, 7, 32, 4, 4, name='conv1', pad=pad, activation=act, bn=False, trainable=trainable) # 56x56
    h = conv(h, 5, 5, 64, 1, 1, name='conv2', pad=pad, activation=act, bn=bn, trainable=trainable) # 52x52
    h = conv(h, 3, 3, 128, 2,2, name='conv3', pad=pad, activation=act, bn=bn, trainable=trainable) # 25x25
    h = conv(h, 3, 3, 256, 1,1, name='conv4', pad=pad, activation=act, bn=bn, trainable=trainable) # 23x23
    h = conv(h, 3, 3, 256, 2,2, name='conv5', pad=pad, activation=act, bn=bn, trainable=trainable) # 11x11
    h = avg_pool(h, 11, 11, 11, 11, name='pool5', pad='VALID')
    h_0 = tf.reshape(h, [-1, 256], name='pool5_reshape')    # 256 from image

    h = fc(feature, 1024, name='feat_fc1', activation=act, bn=bn, trainable=trainable)
    h_1 = fc(h, 512, name='feat_fc2', activation=act, bn=bn, trainable=trainable)   # 512 from feature

    h = tf.concat(axis=1, values=[h_0, h_1], name='concat_fc5')  # 256+512=768
    if trainable:  # train phase
        h = tf.nn.dropout(h, 0.5, name='drop5')
        h = fc(h, 512, name='fc6', activation=act, bn=bn, trainable=trainable)
        h = tf.nn.dropout(h, 0.5, name='drop6')
        h = fc(h, 2, name='fc7', activation='', bn=False, trainable=trainable)
    else:               # test phase
        h = fc(h, 512, name='fc6', activation=act, bn=bn, trainable=trainable)
        h = fc(h, 1, name='fc7', activation='', bn=False, trainable=trainable)
    return h         # range (-inf, inf)


def test():
    net = deepSimNet()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./tmp/', sess.graph)
        writer.flush()


if __name__ == '__main__':
    test()
