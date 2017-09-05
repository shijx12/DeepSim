import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
import os.path as osp
import glob
import json
from tqdm import tqdm

this_dir = osp.dirname(__file__)
print(this_dir)

from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg
from lib.fast_rcnn.test import im_detect
from lib.fast_rcnn.nms_wrapper import nms
from lib.utils.timer import Timer

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


# CLASSES = ('__background__','person','bike','motorbike','car','bus')

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default='0', type=str)
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default=' ')
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    if args.model == ' ' or not os.path.exists(args.model):
        print ('current path is ' + os.path.abspath(__file__))
        raise IOError(('Error: Model not found.\n'))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, 'bbox')):
        os.makedirs(os.path.join(args.output_dir, 'bbox'))

    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network(args.demo_net)
    # load model
    print ('Loading network {:s}... '.format(args.demo_net)),
    saver = tf.train.Saver()
    saver.restore(sess, args.model)
    # saver.restore(sess, tf.train.latest_checkpoint(args.model))
    print (' done.')

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    cls_ind = 15    # person
    
    im_names = os.listdir(args.input_dir)
    person_boxes = {}
    for im_name in tqdm(im_names):
        # Load the demo image
        im = cv2.imread(os.path.join(args.input_dir, im_name))

        # Detect all object classes and regress object bounds
        scores, boxes = im_detect(sess, net, im)

        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        person_boxes[im_name] = []
        for i in keep:
            if dets[i, -1] >= CONF_THRESH:
                person_boxes[im_name].append(map(int, dets[i, :-1].tolist()))
        for box in person_boxes[im_name]:
            cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0, 0, 255))
        cv2.imwrite(os.path.join(args.output_dir, 'bbox', im_name), im)
    json.dump(person_boxes, open(os.path.join(args.output_dir, 'bbox_info.json'), 'w'))



