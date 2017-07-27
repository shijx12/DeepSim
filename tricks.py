
# to find library in ..
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# in TFFRCNN, use entire image as receptive field of ROI, via hack_roi layer
# im_info: a list of [image_height, image_width, scale_ratios]
self.layers['hack_roi'] = tf.py_func(
    lambda im_info: np.array([[0, 0, 0, im_info[0][1], im_info[0][0]]], dtype=np.float32), # [[image_index, x0,y0,x1,y1]] 
    [self.im_info], 
    tf.float32, name='hack_roi')

