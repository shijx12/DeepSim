import argparse
import pprint
import numpy as np
import sys
import os.path
import cv2
import time
import tensorflow as tf

from lib.fast_rcnn.train import get_training_roidb
from lib.fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_log_dir
from lib.datasets.factory import get_imdb
from lib.networks.forked_VGGnet import forked_VGGnet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='GPU device id [0]', default='0', type=str)
    parser.add_argument('--iters', help='number of iterations to train', default=50000, type=int)
    parser.add_argument('--det_branch', dest='det_branch_path', help='well-trained VGG16 object detection model weights path', default=None, type=str, required=True)
    parser.add_argument('--imdb', dest='imdb_name', help='dataset to train on', default='voc_2012_train', type=str)
    parser.add_argument('--seed', dest='seed', type=int, default=123456789)
    parser.add_argument('--output', type=str, required=True, help='output dir to store checkpoints and save results')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of optimizer')
    parser.add_argument('--save_freq', type=int, default=10000, help='save frequency')
    parser.add_argument('--show_freq', type=int, default=50, help='show frequency')
    parser.add_argument('--summ_freq', type=int, default=200, help='summary frequenc')

    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--optimizer', dest='optimizer', type=str)
    args = parser.parse_args()
    return args

class DataFetcher:
    def __init__(self):
        imdb = get_imdb(args.imdb_name)
        # Ignore the background class!!! So ['gt_classes'] must minus 1.
        # We get the classes as the placeholder filler of forked_VGGnet.classes
        self.classes = [ np.zeros(imdb.num_classes - 1) for i in range(imdb.num_images) ]
        for i, anno in enumerate(imdb.gt_roidb()):
            np.put(self.classes[i], map(lambda x: x-1, anno['gt_classes']), 1)
        self.images = [ imdb.image_path_at(i) for i in range(imdb.num_images) ]
        assert len(self.classes) == len(self.images)

        self._perm = np.random.permutation(np.arange(len(self.images)))
        self._cur = 0

    def nextbatch(self):
        # if all images have been trained, permuate again.
        if self._cur >= len(self.images):
            self._cur = 0
            self._perm = np.random.permutation(np.arange(len(self.images)))
        i = self._perm[self._cur]
        self._cur += 1
        blobs = {}
        # substract PIXEL_MEANS from original image.
        im = cv2.imread(self.images[i]).astype(np.float32, copy=False)
        im -= cfg.PIXEL_MEANS
        blobs['data'] = [im]
        blobs['classes'] = [self.classes[i]]
        blobs['keep_prob'] = 0.5 # not used at all
        # im_scale=1, that is, we don't scale the original image size.
        blobs['im_info'] = np.asarray([[im.shape[1], im.shape[2], 1]], dtype=np.float32)
        return blobs


if __name__ == '__main__':
    args = parse_args()
    print(args)
    if not os.path.exists(args.det_branch_path):
        print('Detection branch model path doesnt exist!')
        sys.exit(0)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    np.random.seed(args.seed)
    print('using config:')
    pprint.pprint(cfg)

    
    net = forked_VGGnet(trainable=True) # Net
    data = DataFetcher()    # Data
    sess = tf.Session() # Session
    
    print('training variables:')
    for var in net.cls_variables:
        print(var)
    # Optimizer and train op
    optimizer = tf.train.AdamOptimizer(args.lr, beta1=args.beta1)
    grads = optimizer.compute_gradients(net.cls_loss, net.cls_variables)
    train_op = optimizer.apply_gradients(grads)
    # global step
    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)


    print('Initializing net, saver and tf...')
    sess.run(tf.global_variables_initializer())
    # restore the original detection VGG16 weights
    saver = tf.train.Saver(net.det_variables)
    saver.restore(sess, tf.train.latest_checkpoint(args.det_branch_path))

    # saver of the entire forked_VGGnet
    saver = tf.train.Saver(max_to_keep=None)
    ckpt = tf.train.get_checkpoint_state(args.output)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("forked_VGGnet Model restored..")

    # summary information and handler
    tf.summary.scalar('cls_loss', net.cls_loss)
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(args.output, sess.graph)


    # tf progress initialization
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    tic = time.time()
    
    try:
        for step in range(1, args.iters+1):
            blobs = data.nextbatch()
            feed_dict = {
                net.data: blobs['data'],
                net.im_info: blobs['im_info'],
                net.keep_prob: blobs['keep_prob'],
                net.classes: blobs['classes'],
            }
            # construct run dict
            run_dict = {
                'global_step': global_step,
                'incr_global_step': incr_global_step,
                'train_op': train_op,
            }
            if step % args.show_freq == 0:
                run_dict['cls_loss'] = net.cls_loss
            if step % args.summ_freq == 0:
                run_dict['summary'] = summary_op

            # one step training.
            results = sess.run(run_dict, feed_dict=feed_dict)

            # save, summary and display
            if step % args.show_freq == 0:
                rate = step / (time.time() - tic)
                remaining = (args.iters+1-step) / rate
                print(' step %6d , cls_loss: %6f , remaining %5dm' % (results['global_step'], results['cls_loss'], remaining/60)) 
            if step % args.save_freq == 0:
                print('================ saving model =================')
                saver.save(sess, os.path.join(args.output, 'model'), global_step=results['global_step'])
            if step % args.summ_freq == 0:
                print('-------------- recording summary --------------')
                summary_writer.add_summary(results['summary'], results['global_step'])
    except KeyboardInterrupt:
        print('End Training...')
    finally:
        coord.request_stop()
        coord.join(threads)

