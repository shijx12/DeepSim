import argparse
import pprint
import numpy as np
import sys
import os.path
import time
import math
import tensorflow as tf
from util import DataFetcher
from tqdm import tqdm


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
    parser.add_argument('--summ_freq', type=int, default=100, help='summary frequenc')
    parser.add_argument('--mode', default='train', choices=['train', 'test'])
    parser.add_argument('--debug', action='store_true')

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
    data = DataFetcher(args.imdb_name)    # Data
    sess = tf.Session() # Session
    print('trainable variables:')
    for var in net.trainable_variables:
        print(var)
    # Optimizer and train op
    optimizer = tf.train.AdamOptimizer(args.lr, beta1=args.beta1)

    #one_cls_loss = tf.losses.sigmoid_cross_entropy(net.classes[0][0], net.layers['forked_cls_score'][0][0])
    #grads = optimizer.compute_gradients(one_cls_loss, net.trainable_variables)  # TODO: exp

    grads = optimizer.compute_gradients(net.cls_loss, net.trainable_variables)
    train_op = optimizer.apply_gradients(grads)
    # global step
    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    print('Initializing net, saver and tf...')
    sess.run(tf.global_variables_initializer())
    # restore the original detection VGG16 weights
    if args.det_branch_path:
        #saver = tf.train.Saver(net.det_variables)
        #saver.restore(sess, tf.train.latest_checkpoint(args.det_branch_path))
        net.load(args.det_branch_path, sess, True)

    # saver of the entire forked_VGGnet
    saver = tf.train.Saver(max_to_keep=None)
    ckpt = tf.train.get_checkpoint_state(args.logdir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("forked_VGGnet Model restored..")

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
                run_dict['cls_loss'] = net.cls_loss
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
    data = DataFetcher(args.imdb_name)


    def check_sparsity():
        (net.feed('conv5_3', 'hack_roi')
                .roi_pool(7, 7, 1.0/16, name='pool_5')
                .fc(4096, name='fc6', trainable=False)
                .fc(4096, name='fc7', trainable=False)) # NOTE: must comment VGGnet_test's corresponding part
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(args.logdir)
        saver.restore(sess, ckpt.model_checkpoint_path)
        names = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3', 'conv5_3', 'fc6', 'fc7', 'forked_fc6', 'forked_fc7']
        layers = { name: tf.nn.zero_fraction(net.get_output(name)) for name in names }
        def next():
            blob = data.nextbatch()
            feed_dict = {net.data:blob['data'], net.im_info:blob['im_info'], net.keep_prob:0.5, net.classes:blob['classes']}
            results = sess.run(layers, feed_dict=feed_dict)
            for name, result in results.items():
                print name, result 
        from IPython import embed; embed()
        sys.exit(0)
    # check_sparsity()


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
        from IPython import embed; embed()
        print('target_pos_num\ttrue_pos\tprecision\trecall')
        for i in range(20):
            print('%.0f\t%.0f\t%.2f\t%.2f' % (target_pos_num[i], true_pos[i], precision[i], recall[i]))
    #cal_prec_reca_for_each_class()


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()

