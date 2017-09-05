import tensorflow as tf
import numpy as np
import argparse
import time
import math
from tqdm import tqdm
import cfg
import os
from util import *

class Encoder_net():
    def __init__(self, trainable=True):
        self.original_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='original_image')
        self.image = subtract_mean(crop(self.original_image, cfg.RESIZED_SIZE, cfg.IMAGE_SIZE))  # crop to fixed size and subtract
        self.classes = tf.placeholder(tf.float32, shape=[None, cfg.N_CLASSES])

        h = conv(self.image, 3, 3, 64, 1, 1, name='conv1_1', trainable=False)
        h = conv(h, 3, 3, 64, 1, 1, name='conv1_2', trainable=False)
        h = max_pool(h, 2, 2, 2, 2, pad='VALID', name='pool1')
        h = conv(h, 3, 3, 128, 1, 1, name='conv2_1', trainable=False)
        h = conv(h, 3, 3, 128, 1, 1, name='conv2_2', trainable=False)
        h = max_pool(h, 2, 2, 2, 2, pad='VALID', name='pool2')
        h = conv(h, 3, 3, 256, 1, 1, name='conv3_1', trainable=trainable)
        h = conv(h, 3, 3, 256, 1, 1, name='conv3_2', trainable=trainable)
        h = conv(h, 3, 3, 256, 1, 1, name='conv3_3', trainable=trainable)
        h = max_pool(h, 2, 2, 2, 2, pad='VALID', name='pool3')
        h = conv(h, 3, 3, 512, 1, 1, name='conv4_1', trainable=trainable)
        h = conv(h, 3, 3, 512, 1, 1, name='conv4_2', trainable=trainable)
        h = conv(h, 3, 3, 512, 1, 1, name='conv4_3', trainable=trainable)
        h = max_pool(h, 2, 2, 2, 2, pad='VALID', name='pool4')
        h = conv(h, 3, 3, 512, 1, 1, name='conv5_1', trainable = trainable)
        h = conv(h, 3, 3, 512, 1, 1, name='conv5_2', trainable = trainable)
        h = conv(h, 3, 3, 512, 1, 1, name='conv5_3', trainable = trainable) # 14x14x512
        h = max_pool(h, 2, 2, 2, 2, pad='VALID', name='pool5') # 7x7x512
        h = tf.reshape(h, [-1, 7*7*512], name='reshape_pool5')
        if trainable:
            h = fc(h, 4096, name='fc6', trainable=trainable)
            h = tf.nn.dropout(h, 0.5, name='drop6')
            h = fc(h, 4096, name='fc7', trainable=trainable)
            h = tf.nn.dropout(h, 0.5, name='drop7')
        else:
            h = fc(h, 4096, name='fc6', trainable=trainable)
            h = fc(h, 4096, name='fc7', trainable=trainable)
        self.outputs = fc(h, cfg.N_CLASSES, activation='', name='cls_score', trainable=trainable)

        # Classification loss.
        self.cls_loss = tf.losses.sigmoid_cross_entropy(self.classes, self.outputs)
        self.outputs = tf.nn.sigmoid(self.outputs)

        # Variable collector.
        self.restore_variables = [var for var in tf.all_variables() if not var.name.startswith('cls_score')]
        self.trainable_variables = tf.trainable_variables()

    def load(self, data_path, session, ignore_missing=False):
        data_dict = np.load(data_path).item()
        for key in data_dict:
            with tf.variable_scope(key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        session.run(var.assign(data_dict[key][subkey]))
                        print 'assign pretrain model '+subkey+' to '+key
                    except ValueError:
                        print 'ignore '+key
                        if not ignore_missing:
                            raise

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
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
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
    data = DataFetcher(args.imdb_name)
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
    data = DataFetcher(args.imdb_name)
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


