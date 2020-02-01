import numpy as np
import cv2
from PIL import Image
from rknn.api import RKNN
from timeit import default_timer as timer

GRID0 = 13
GRID1 = 26
LISTSIZE = 85
SPAN = 3
NUM_CLS = 80
MAX_BOXES = 500
OBJ_THRESH = 0.2
NMS_THRESH = 0.2

CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def process(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2])
    box_wh = np.exp(input[..., 2:4])
    box_wh = box_wh * anchors

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)

    box_xy += grid
    box_xy /= (grid_w, grid_h)
    box_wh /= (416, 416)
    box_xy -= (box_wh / 2.)
    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= OBJ_THRESH)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov3_post_process(input_data):
    # # yolov3
    # masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
    #            [59, 119], [116, 90], [156, 198], [373, 326]]
    # yolov3-tiny
    masks = [[3, 4, 5], [0, 1, 2]]
    anchors = [[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]]

    boxes, classes, scores = [], [], []
    for input,mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    # # Scale boxes back to original image shape.
    # width, height = 416, 416 #shape[1], shape[0]
    # image_dims = [width, height, width, height]
    # boxes = boxes * image_dims

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(x, y, x+w, y+h))
        x *= image.shape[1]
        y *= image.shape[0]
        w *= image.shape[1]
        h *= image.shape[0]
        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

        # print('class: {0}, score: {1:.2f}'.format(CLASSES[cl], score))
        # print('box coordinate x,y,w,h: {0}'.format(box))

def load_model():
        rknn = RKNN()
        print('-->loading model')
        rknn.load_rknn('./yolov3_tiny.rknn')
        #rknn.load_rknn('./yolov3.rknn')
        print('loading model done')

        print('--> Init runtime environment')
        ret = rknn.init_runtime()
        if ret != 0:
                print('Init runtime environment failed')
                exit(ret)
        print('done')
        return rknn

def predict(rknn):
        im = Image.open("./data/dog.jpg")
        im = im.resize((416, 416))
        #im = im.resize((608, 608))
        mat = np.asarray(im.convert('RGB'))
        out_boxes, out_boxes2, out_boxes3 = rknn.inference(inputs=[mat])
        print(len(out_boxes[0].tolist()))
        print(len(out_boxes2[0].tolist()))
        print(len(out_boxes3[0].tolist()))
if __name__ == '__main__':
    rknn = load_model()
    font = cv2.FONT_HERSHEY_SIMPLEX;
    capture = cv2.VideoCapture("data/3.mp4")
    #capture = cv2.VideoCapture(0)
    accum_time = 0
    curr_fps = 0
    prev_time = timer()
    fps = "FPS: ??"
    while(True):
        ret, frame = capture.read()
        if ret == True:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.resize(frame, (416, 416))

            testtime=timer()
            out_boxes, out_boxes2 = rknn.inference(inputs=[image])
            testtime2=timer()
            print("rknn use time {}", testtime2-testtime)

            out_boxes = out_boxes.reshape(SPAN, LISTSIZE, GRID0, GRID0)
            out_boxes2 = out_boxes2.reshape(SPAN, LISTSIZE, GRID1, GRID1)
            input_data = []
            input_data.append(np.transpose(out_boxes, (2, 3, 0, 1)))
            input_data.append(np.transpose(out_boxes2, (2, 3, 0, 1)))
            
            testtime=timer()
            boxes, classes, scores = yolov3_post_process(input_data)
            testtime2=timer()
            print("process use time: {}", testtime2-testtime)
            
            testtime=timer()
            if boxes is not None:
                draw(image, boxes, scores, classes)
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time += exec_time
            curr_fps += 1
            if accum_time > 1:
                accum_time -= 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(image, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                         fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.imshow("results",image)
            c = cv2.waitKey(5) & 0xff
            if c == 27:
                cv2.destroyAllWindows()
                capture.release()
                break;
            testtime2=timer()
            print("show image use time: {}", testtime2-testtime)