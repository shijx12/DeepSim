import argparse
import pprint
import numpy as np
import sys
import os.path
import cv2
import time
import math
import tensorflow as tf
from tqdm import tqdm

from lib.fast_rcnn.train import get_training_roidb
from lib.fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_log_dir
from lib.datasets.factory import get_imdb
from lib.networks.forked_VGGnet import forked_VGGnet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='GPU device id [0]', default='0', type=str)
    parser.add_argument('--iters', help='number of iterations to train', default=50000, type=int)
    parser.add_argument('--det_branch', dest='det_branch_path', help='well-trained VGG16 object detection model weights path', default=None, type=str)
    parser.add_argument('--conv1_5_trainable', type=str, default='00000', help='whether conv1 to conv5 is trainable. 1 is trainable and 0 is not. default is 00000')
    parser.add_argument('--imdb', dest='imdb_name', help='dataset to train on', default='voc_2012_train', type=str)
    parser.add_argument('--seed', dest='seed', type=int, default=123456789)
    parser.add_argument('--logdir', type=str, required=True, help='log dir to store checkpoints and save results')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of optimizer')
    parser.add_argument('--save_freq', type=int, default=10000, help='save frequency')
    parser.add_argument('--show_freq', type=int, default=100, help='show frequency')
    parser.add_argument('--summ_freq', type=int, default=200, help='summary frequenc')
    parser.add_argument('--mode', default='train', choices=['train', 'test'])
    parser.add_argument('--debug', action='store_true')
    
    parser.add_argument('--which_class', type=int, default=0)

    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--optimizer', dest='optimizer', type=str)
    args = parser.parse_args()
    return args
args = parse_args()

class DataFetcher:
    def __init__(self, which_class=-1):
        imdb = get_imdb(args.imdb_name)
        # Ignore the background class!!! So ['gt_classes'] must minus 1.
        # We get the classes as the placeholder filler of forked_VGGnet.classes
        self.classes = [ np.zeros(imdb.num_classes - 1) for i in range(imdb.num_images) ]
        for i, anno in enumerate(imdb.gt_roidb()):
            np.put(self.classes[i], map(lambda x: x-1, anno['gt_classes']), 1)
        self.images = [ imdb.image_path_at(i) for i in range(imdb.num_images) ]

        if which_class >= 0:
            pos_num = 0
            for cls in self.classes:
                if cls[which_class] == 1:
                    pos_num += 1
            print('postive number of class%d is %d' % ( which_class, pos_num ))
            neg_num = 0
            sub_classes = []
            sub_images = []
            for i, c in zip(self.images, self.classes):
                if c[which_class] == 1:
                    sub_classes.append(c)
                    sub_images.append(i)
                elif neg_num < pos_num:
                    sub_classes.append(c)
                    sub_images.append(i)
                    neg_num += 1
            self.classes = sub_classes
            self.images = sub_images
            print('negtive number of class%d is %d' % ( which_class, neg_num ))
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
        blobs['path'] = [self.images[i]]
        blobs['classes'] = [self.classes[i]]
        blobs['keep_prob'] = 0.5 # not used at all
        # im_info: a list of [image_height, image_width, scale_ratios]
        # im_scale=1, that is, we don't scale the original image size.
        blobs['im_info'] = np.asarray([[im.shape[1], im.shape[2], 1]], dtype=np.float32)
        return blobs


def train():
    print(args)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    if len(args.conv1_5_trainable) != 5 or np.any([ c not in ['0', '1'] for c in args.conv1_5_trainable ]):
        raise Exception('Invalid argument conv1_5_trainable. length must equal to 5 and composed with [0, 1]')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    #print('using config:')
    #pprint.pprint(cfg)

    conv1_5_trainable = [ c=='1' for c in args.conv1_5_trainable ]
    net = forked_VGGnet(True, conv1_5_trainable) # Net
    data = DataFetcher(args.which_class)    # Data
    sess = tf.Session() # Session
    print('trainable variables:')
    for var in net.trainable_variables:
        print(var)
    # Optimizer and train op
    optimizer = tf.train.AdamOptimizer(args.lr, beta1=args.beta1)

    one_cls_loss = tf.losses.sigmoid_cross_entropy(net.classes[0][args.which_class], net.layers['forked_cls_scores'][0][args.which_class])
    grads = optimizer.compute_gradients(one_cls_loss, net.trainable_variables)  # TODO: exp

    # grads = optimizer.compute_gradients(net.cls_loss, net.trainable_variables)
    train_op = optimizer.apply_gradients(grads)
    # global step
    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    print('Initializing net, saver and tf...')
    sess.run(tf.global_variables_initializer())
    # restore the original detection VGG16 weights
    if args.det_branch_path:
        saver = tf.train.Saver(net.det_variables)
        saver.restore(sess, tf.train.latest_checkpoint(args.det_branch_path))

    # saver of the entire forked_VGGnet
    saver = tf.train.Saver(max_to_keep=None)
    ckpt = tf.train.get_checkpoint_state(args.logdir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("forked_VGGnet Model restored..")

    # summary information and handler
    tf.summary.scalar('class%d_loss' % args.which_class, one_cls_loss)
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(args.logdir, sess.graph)


    # tf process initialization
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
                # net.classes: np.ones(np.asarray(blobs['classes']).shape) # for toy training
            }
            # construct run dict
            run_dict = {
                'global_step': global_step,
                'incr_global_step': incr_global_step,
                'train_op': train_op,
            }
            if step % args.show_freq == 0:
                run_dict['cls_loss'] = one_cls_loss
            if step % args.summ_freq == 0:
                run_dict['summary'] = summary_op
            if args.debug:
                run_dict['outputs'] = net.outputs

            # one step training.
            results = sess.run(run_dict, feed_dict=feed_dict)

            # save, summary and display
            if step % args.show_freq == 0:
                rate = step / (time.time() - tic)
                remaining = (args.iters+1-step) / rate
                print(' step %6d , cls_loss: %6f , remaining %5dm' % (results['global_step'], results['cls_loss'], remaining/60)) 
                if args.debug:
                    for i in range(20):
                        print blobs['classes'][0][i], math.floor(10*results['outputs'][0][i])
            if step % args.save_freq == 0:
                print('================ saving model =================')
                saver.save(sess, os.path.join(args.logdir, 'model'), global_step=results['global_step'])
            if step % args.summ_freq == 0:
                print('-------------- recording summary --------------')
                summary_writer.add_summary(results['summary'], results['global_step'])
    except KeyboardInterrupt:
        print('End Training...')
    finally:
        coord.request_stop()
        coord.join(threads)

def test():
    if not os.path.exists(args.logdir):
        raise Exception('log dir doesnt exist!')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    net = forked_VGGnet(train=False)
    data = DataFetcher()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(args.logdir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("forked_VGGnet Model restored..")
    #from IPython import embed; embed()
    
    def cal_prec_reca_for_each_class():
        targets = []
        outputs = []
        paths = []
        for i in tqdm(range(len(data.images))):
            blob = data.nextbatch()
            feed_dict = {net.data:blob['data'], net.im_info:blob['im_info'], net.keep_prob:0.5, net.classes:blob['classes']}
            output = sess.run(net.outputs, feed_dict=feed_dict)
            targets.append(blob['classes'][0])
            outputs.append(output[0])
            paths.append(blob['path'][0])
        thres = 0.5
        binary_output = [[1 if o >= thres else 0 for o in output] for output in outputs]
        true_pos = np.zeros(20)
        tp_paths_idx = [[] for i in range(20)]
        fn_paths_idx = [[] for i in range(20)]
        for i in range(len(targets)):
            for j in range(20):
                if targets[i][j] == 1 and binary_output[i][j] == 1:
                    true_pos[j] += 1
                    tp_paths_idx[j].append(i)
                elif targets[i][j] == 1 and binary_output[i][j] != 1:
                    fn_paths_idx[j].append(i)
        target_pos_num = np.sum(targets, axis=0)
        output_pos_num = np.sum(binary_output, axis=0)
        precision = true_pos / output_pos_num
        recall = true_pos / target_pos_num
        print('target_pos_num\ttrue_pos\tprecision\trecall')
        for i in range(20):
            print('%.0f\t\t%.0f\t%.2f\t%.2f' % (target_pos_num[i], true_pos[i], precision[i], recall[i]))
        from IPython import embed; embed()
    cal_prec_reca_for_each_class()


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()

