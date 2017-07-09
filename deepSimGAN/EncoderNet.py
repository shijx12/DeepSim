import tensorflow as tf
import numpy as np
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # to find ..lib
import time
import math
from tqdm import tqdm
from lib.networks.network import Network
import cfg
import util

class Encoder_net(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.original_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='original_image')
        self.image = util.subtract_mean(util.crop(self.original_image, cfg.RESIZED_SIZE, cfg.HEIGHT))  # crop to fixed size and subtract mean pixel value. NOTE: subtract_mean is necessary!
        self.layers = dict({'image': self.image})
        # Note: We don't consider the background class at the forked_VGGnet!!!!!!!
        self.n_classes = cfg.N_CLASSES
        self.classes = tf.placeholder(tf.float32, shape=[None, self.n_classes])
        self.trainable = trainable
        self.setup()
        
        # im_info: a list of [image_height, image_width, scale_ratios]
        #self.layers['hack_roi'] = tf.py_func(
        #    lambda im_info: np.array([[0, 0, 0, im_info[0][1], im_info[0][0]]], dtype=np.float32), # [[image_index, x0,y0,x1,y1]] 
        #    [self.im_info], 
        #    tf.float32, name='hack_roi')

    def setup(self):
        (self.feed('image')
             .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
             .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
             .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1', trainable=self.trainable)
             .conv(3, 3, 256, 1, 1, name='conv3_2', trainable=self.trainable)
             .conv(3, 3, 256, 1, 1, name='conv3_3', trainable=self.trainable)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1', trainable=self.trainable)
             .conv(3, 3, 512, 1, 1, name='conv4_2', trainable=self.trainable)
             .conv(3, 3, 512, 1, 1, name='conv4_3', trainable=self.trainable)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1', trainable = self.trainable)
             .conv(3, 3, 512, 1, 1, name='conv5_2', trainable = self.trainable)
             .conv(3, 3, 512, 1, 1, name='conv5_3', trainable = self.trainable) # 14x14x512
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool5'))  # 7x7x512
        if self.trainable:   # for training phase
            (self.feed('pool5')
                .fc(4096, name='fc6', trainable=True)
                .dropout(0.5, name='drop6')
                .fc(4096, name='fc7', trainable=True)
                .dropout(0.5, name='drop7')
                .fc(self.n_classes, activation=None, name='cls_score', trainable=True))
        else:   # for test phase, just remove the dropout layer.
            (self.feed('pool5')
                .fc(4096, name='fc6', trainable=False)
                .fc(4096, name='fc7', trainable=False)
                .fc(self.n_classes, activation=None, name='cls_score', trainable=False))

        # Classification loss.
        self.outputs = self.layers['cls_score']
        self.cls_loss = tf.losses.sigmoid_cross_entropy(self.classes, self.outputs)
        self.outputs = tf.nn.sigmoid(self.outputs)

        # Variable collector.
        self.restore_variables = [var for var in tf.all_variables() if not var.name.startswith('cls_score')]
        self.trainable_variables = tf.trainable_variables()

        for layer in ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'pool5', 'fc6', 'fc7', 'cls_score']:
            h = self.get_output(layer)
            tf.summary.histogram('Activation/'+h.name, h)
            tf.summary.scalar('Sparsity/'+h.name, tf.nn.zero_fraction(h))

def visual():
    fvgg = Encoder_net(True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("tmp/", sess.graph)
    writer.flush()







def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='GPU device id [0]', default='0', type=str)
    parser.add_argument('--iters', help='number of iterations to train', default=50000, type=int)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--weight_path',  help='well-trained VGG16 model weights path', default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name', help='dataset to train on', default='voc_2012_train', type=str)
    parser.add_argument('--seed', dest='seed', type=int, default=123456789)
    parser.add_argument('--logdir', type=str, required=True, help='log dir to store checkpoints and save results')
    parser.add_argument('--restore_step', type=str, help='restore step for test mode')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of optimizer')
    parser.add_argument('--save_freq', type=int, default=10000, help='save frequency')
    parser.add_argument('--show_freq', type=int, default=100, help='show frequency')
    parser.add_argument('--summ_freq', type=int, default=100, help='summary frequenc')
    parser.add_argument('--mode', default='train', choices=['train', 'test'])
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--optimizer', dest='optimizer', type=str)
    args = parser.parse_args()
    return args
args = parse_args()

def train():
    print(args)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    net = Encoder_net(True)
    data = util.DataFetcher(args.imdb_name)
    sess = tf.Session()

    print('trainable variables:')
    for var in net.trainable_variables:
        print(var)
    # Optimizer and train op
    optimizer = tf.train.AdamOptimizer(args.lr, beta1=args.beta1)
    grads = optimizer.compute_gradients(net.cls_loss, net.trainable_variables)
    train_op = optimizer.apply_gradients(grads)
    # global step
    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    print('Initializing net, saver and tf...')
    sess.run(tf.global_variables_initializer())
    # restore the original detection VGG16 weights
    if args.weight_path:
        #saver = tf.train.Saver(net.restore_variables)
        #saver.restore(sess, tf.train.latest_checkpoint(args.weight_path))  # For object detection weight, tf checkpoint file
        net.load(args.weight_path, sess, True)  # For imagenet weight, npy file

    # saver of the entire forked_VGGnet
    saver = tf.train.Saver(max_to_keep=None)
    ckpt = tf.train.get_checkpoint_state(args.logdir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored..")

    # summary information and handler
    tf.summary.scalar('cls_loss', net.cls_loss)
    for grad, var in grads:
        tf.summary.histogram(var.name, var)
        tf.summary.histogram(var.name+'/grad', grad)
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(args.logdir, sess.graph)

    # tf process initialization
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    tic = time.time()
    
    try:
        for step in range(1, args.iters+1):
            blobs = data.nextbatch(args.batch_size)
            feed_dict = {
                net.original_image: blobs['data'],
                net.classes: blobs['classes'],
            }
            run_dict = {
                'global_step': global_step,
                'incr_global_step': incr_global_step,
                'train_op': train_op,
            }
            if step % args.show_freq == 0:
                run_dict['cls_loss'] = net.cls_loss
            if step % args.summ_freq == 0:
                run_dict['summary'] = summary_op
            if args.debug:
                run_dict['outputs'] = net.outputs

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
    
    net = Encoder_net(False)
    data = util.DataFetcher(args.imdb_name)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(args.logdir)
    if ckpt and ckpt.model_checkpoint_path:
        if args.restore_step:
            saver.restore(sess, os.path.join(args.logdir, 'model-'+args.restore_step))
        else:
            saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored..")
    #from IPython import embed; embed() 

    def cal_prec_reca_for_each_class():
        targets = []
        outputs = []
        paths = []
        for i in tqdm(range(len(data.images))):
            blob = data.nextbatch()
            feed_dict = {net.original_image:blob['data'], net.classes:blob['classes']}
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
        print('target_pos_num\ttrue_pos\tprecision\trecall\tf1')
        for i in range(20):
            print('%.0f\t\t%.0f\t\t%.2f\t\t%.2f\t\t%.2f' % (target_pos_num[i], true_pos[i], precision[i], recall[i], 2*precision[i]*recall[i]/(precision[i]+recall[i])))
        from IPython import embed; embed()
    cal_prec_reca_for_each_class()


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()


