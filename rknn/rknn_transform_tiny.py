from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

import re
import math
import random

from rknn.api import RKNN


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

    # Load tensorflow model
    print('--> Loading model')
    #rknn.load_darknet(model='./yolov3.cfg', weight="./yolov3.weights")
    rknn.load_darknet(model='./yolov3-tiny.cfg', weight="./yolov3-tiny.weights")

    print('done')

    rknn.config(channel_mean_value='0 0 0 255', reorder_channel='0 1 2')

    # Build model
    print('--> Building model')
    rknn.build(do_quantization=True, dataset='./dataset.txt')
    #rknn.build(do_quantization=True, dataset='./dataset_608x608.txt')
    print('done')

    rknn.export_rknn('./yolov3_tiny.rknn')
    #rknn.export_rknn('./yolov3.rknn')

    #rknn.load_rknn('./yolov3.rknn')
    #image = Image.open('./dog.jpg').resize((416, 416))
    #rknn.eval_perf(inputs=[image], is_print=True)

    exit(0)
