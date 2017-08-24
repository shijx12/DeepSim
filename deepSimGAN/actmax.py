import tensorflow as tf
import numpy as np
import argparse
import os
from deepSimNet import Generator
from deepSimNet import Encoder
import cfg

class ActMaxNet():
    def __init__(self):
        self.h = tf.placeholder(tf.float32, shape=[1, cfg.INVERTED_H_DIM], name='inverted_h')
        with tf.name_scope('generator'):
            with tf.variable_scope('generator'):
                self.fake_image = Generator(self.h, trainable=False)
        with tf.name_scope('encoder'):
            self.fake_h, __, self.cls_score = Encoder(self.fake_image, trainable=False) 

        self.gen_variables = [var for var in tf.global_variables() if var.name.startswith('generator')]
        self.enc_variables = [var for var in tf.global_variables() if var not in self.gen_variables]

    def sample(self, label, iters=100): # label [1,20]
        loss = 1-tf.sigmoid(self.cls_score[label-1])
        grad = tf.gradients(loss, self.h)
        apply = self.h - tf.multiply(grad[0], epsilon)

        h_val = np.random(shape=[1, cfg.INVERTED_H_DIM])
        with tf.Session() as sess:
            for i in iters:
                h_val, grad_val = sess.run([apply, grad], feed_dict={self.h, h_val})



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--encoder', type=str, required=True)
    parser.add_argument('--generator', type=str, required=True)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--iters', default=200, type=int)
    parser.add_argument('--eps', nargs=3, type=float, required=True)
    parser.add_argument('--seed', type=int, default=123456789)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    return args
args = parse_args()



def main():
    print(args)
    logdir = os.path.join(args.logdir, 'eps1_%f_eps2_%f_eps3_%f' % (args.eps[0], args.eps[1], args.eps[2]))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    with open(os.path.join(logdir, 'args'), 'w') as f:
        for k, v in vars(args).items():
            f.write(k+':'+str(v)+'\n')

    net = ActMaxNet()
    sess = tf.Session()

