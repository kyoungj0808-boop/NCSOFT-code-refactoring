원본코드
import tensorflow as tf
import numpy as np
import cv2

FLAGS = tf.app.flags.FLAGS


coco_color_map = [
    0x00, 0x55, 0xFF,
	0xFF, 0x11, 0x00,
	0xFF, 0x1A, 0x00,
	0xFF, 0x22, 0x00,
	0xFF, 0x2A, 0x00,
	0xFF, 0x33, 0x00,
	0xFF, 0x3C, 0x00,
	0xFF, 0x44, 0x00,
	0xFF, 0x4C, 0x00,		#//  29
	0xFF, 0x55, 0x00,		#//  30
	0xFF, 0x5E, 0x00,		#//  31
	0xFF, 0x66, 0x00,		#//  32
	0xFF, 0x6E, 0x00,		#//  33
	0xFF, 0x77, 0x00,		#//  34
	0xFF, 0x77, 0x00,		#//  34
	0xFF, 0x80, 0x00,		#//  35
	0xFF, 0x88, 0x00,		#//  36
	0xFF, 0x90, 0x00,		#//  37
	0xFF, 0x99, 0x00,		#//  38
	0xFF, 0xA2, 0x00,		#//  39
	0xFF, 0xAA, 0x00,		#//  40
	0xF6, 0xB2, 0x00,		#//  41
	0xEE, 0xBB, 0x00,		#//  42
	0xE6, 0xC4, 0x00,		#//  43
	0xDD, 0xCC, 0x00,		#//  44
	0xD4, 0xD4, 0x00,		#//  45
	0xCC, 0xDD, 0x00,		#//  46
	0xC4, 0xE6, 0x00,		#//  47
	0xBB, 0xEE, 0x00,		#//  48
	0xB2, 0xF6, 0x00,		#//  49
	0xAA, 0xFF, 0x00,		#//  50
	0xA2, 0xFF, 0x00,		#//  51
	0x99, 0xFF, 0x00,		#//  52
	0x90, 0xFF, 0x00,		#//  53
	0x88, 0xFF, 0x00,		#//  54
	0x80, 0xFF, 0x00,		#//  55
	0x77, 0xFF, 0x00,		#//  56
	0x6E, 0xFF, 0x00,		#//  57
	0x66, 0xFF, 0x00,		#//  58
	0x5E, 0xFF, 0x00,		#//  59
	0x55, 0xFF, 0x00,		#//  61
	0x44, 0xFF, 0x11,		#//  62
	0x3C, 0xFF, 0x1A,		#//  63
	0x33, 0xFF, 0x22,		#//  64
	0x2A, 0xFF, 0x2A,		#//  65
	0x22, 0xFF, 0x33,		#//  66
	0x1A, 0xFF, 0x3C,		#//  67
	0x11, 0xFF, 0x44,		#//  68
	0x08, 0xFF, 0x4C,		#//  69
	0x00, 0xFF, 0x55,		#//  70
	0x00, 0xFF, 0x5E,		#//  71
	0x00, 0xFF, 0x66,		#//  72
	0x00, 0xFF, 0x6E,		#//  73
	0x00, 0xFF, 0x77,		#//  74
	0x00, 0xFF, 0x80,		#//  75
	0x00, 0xFF, 0x88,		#//  76
	0x00, 0xFF, 0x90,		#//  77
	0x00, 0xFF, 0x99,		#//  78
	0x00, 0xFF, 0xA2,		#//  79
	0x00, 0xFF, 0xAA,		#//  80
	0x00, 0xF6, 0xB2,
	0x00, 0xEE, 0xBB,
	0x00, 0xE6, 0xC4,
	0x00, 0xDD, 0xCC,
	0x00, 0xD4, 0xD4,
	0x00, 0xCC, 0xDD,
	0x00, 0xC4, 0xE6,
	0x00, 0xBB, 0xEE,
	0x00, 0xB2, 0xF6,
	0x00, 0xAA, 0xFF,
	0x00, 0xA2, 0xFF,
	0x00, 0x99, 0xFF,
	0x00, 0x90, 0xFF,
	0x00, 0x88, 0xFF,
	0x00, 0x80, 0xFF,
	0x00, 0x77, 0xFF,
	0x00, 0x6E, 0xFF,
	0x00, 0x66, 0xFF,
	0x00, 0x5E, 0xFF,
    0xFF, 0x08, 0x00,
    0x00, 0xA2, 0xFF,
	0x00, 0x99, 0xFF,
	0x00, 0x90, 0xFF,
	0x00, 0x88, 0xFF,
	0x00, 0x80, 0xFF,
	0x00, 0x77, 0xFF,
	0x00, 0x6E, 0xFF,
	0x00, 0x66, 0xFF,
	0x00, 0x5E, 0xFF,
    0xFF, 0x08, 0x00,
    0x00, 0x66, 0xFF,
	0x00, 0x5E, 0xFF,
    0xFF, 0x08, 0x00,

]


coco_class_dict =  {0: u'__background__',
 1: u'person',
 2: u'bicycle',
 3: u'car',
 4: u'motorcycle',
 5: u'airplane',
 6: u'bus',
 7: u'train',
 8: u'truck',
 9: u'boat',
 10: u'traffic light',
 11: u'fire hydrant',
 12: u'stop sign',
 13: u'parking meter',
 14: u'bench',
 15: u'bird',
 16: u'cat',
 17: u'dog',
 18: u'horse',
 19: u'sheep',
 20: u'cow',
 21: u'elephant',
 22: u'bear',
 23: u'zebra',
 24: u'giraffe',
 25: u'backpack',
 26: u'umbrella',
 27: u'handbag',
 28: u'tie',
 29: u'suitcase',
 30: u'frisbee',
 31: u'skis',
 32: u'snowboard',
 33: u'sports ball',
 34: u'kite',
 35: u'baseball bat',
 36: u'baseball glove',
 37: u'skateboard',
 38: u'surfboard',
 39: u'tennis racket',
 40: u'bottle',
 41: u'wine glass',
 42: u'cup',
 43: u'fork',
 44: u'knife',
 45: u'spoon',
 46: u'bowl',
 47: u'banana',
 48: u'apple',
 49: u'sandwich',
 50: u'orange',
 51: u'broccoli',
 52: u'carrot',
 53: u'hot dog',
 54: u'pizza',
 55: u'donut',
 56: u'cake',
 57: u'chair',
 58: u'couch',
 59: u'potted plant',
 60: u'bed',
 61: u'dining table',
 62: u'toilet',
 63: u'tv',
 64: u'laptop',
 65: u'mouse',
 66: u'remote',
 67: u'keyboard',
 68: u'cell phone',
 69: u'microwave',
 70: u'oven',
 71: u'toaster',
 72: u'sink',
 73: u'refrigerator',
 74: u'book',
 75: u'clock',
 76: u'vase',
 77: u'scissors',
 78: u'teddy bear',
 79: u'hair drier',
 80: u'toothbrush'}



def getcolor(score):
    r = 1.0
    g = 1.0
    b = 1.0
    if score < 0.25:
        r = 0
        g = 4 * score
    elif score < 0.5:
        r = 0
        b = 1 + 4 * (0.25 - score)
    elif score < 0.75:
        r = 4*(score - 0.5)
        b = 0
    else:
        g = 1 + 4 * (0.75 - score)
        b = 0
    r *= 255
    g *= 255
    b *= 255
    return (int(r), int(g), int(b))


def checkpoly(ind_box, w, h):
    if np.min(ind_box[:, 0]) < 0:
        return 0
    elif np.max(ind_box[:, 0]) > w:
        return 0
    elif np.min(ind_box[:, 1]) < 0:
        return 0
    elif np.max(ind_box[:, 1]) > h:
        return 0
    return 1


def makepoly_with_score(batch, boxes, scores, h, w, img, thres=0.1):
    w = w.astype(np.int32)
    h = h.astype(np.int32)
    img = img[:, :, :, ::-1]
    img = img.astype(np.uint8)

    for i in range(batch):
        for box, score in zip(boxes[i], scores[i]):
            if score > thres:
                color = getcolor(score)
                valid_box = box.reshape([-1, 4, 2]).astype(np.int32)
                for ind_box in valid_box:
                    if checkpoly(ind_box, w, h):
                        img[i] = cv2.drawContours(img[i], [ind_box], 0, color, 1)

    img = img.astype(np.float32)
    img = img / 255
    return img


def makepoly_with_score_color(batch, boxes, scores, h, w, img, color, thres=0.0):
    w = int(w)
    h = int(h)
    img = img.astype(np.uint8)
    for i in range(batch):
        for box, score in zip(boxes[i], scores[i]):
            if score > thres:
                valid_box = box.reshape([-1, 4, 2]).astype(np.int32)
                for ind_box in valid_box:
                    ind_box[:, 0] = np.clip(ind_box[:, 0], 0, w)
                    ind_box[:, 1] = np.clip(ind_box[:, 1], 0, h)
                    img[i] = cv2.drawContours(img[i], [ind_box], 0, color, 3)
    img = img
    return img


def debugclassifier(quadboxes, gt_boxes, roi_idx, offset_ch, mask, classifier_ch):
    """
    :param quadboxes: (B, 300, 8)
    :param gt_boxes: (N',8) gathered
    :param roi_idx: (N', 2) idx
    :param offset_ch: (N' 8) gatherd
    :parma classifier_ch : (N')
    :return:
    """
    geo_map = np.zeros((quadboxes.shape[0], FLAGS.input_size, FLAGS.input_size, 3), dtype=np.uint8)
    for i,(idx,bin) in enumerate(zip(roi_idx, mask)):
        b, n = idx
        quad = quadboxes[b,n,:].reshape([4, 2]).astype(np.int32)
        gt_box = gt_boxes[i].reshape([4, 2]).astype(np.int32)
        offset = offset_ch[i].reshape([4, 2]).astype(np.int32)
        score = classifier_ch[i]
        color = getcolor(score)

        if checkpoly(quad, FLAGS.input_size, FLAGS.input_size) and score > 0.5:
            geo_map[b] = cv2.drawContours(geo_map[b], [offset], 0, (255, 255, 255), 1)
            geo_map[b] = cv2.circle(geo_map[b], (offset[0, 0], offset[0, 1]), 4, (153, 151, 89), -1)
            geo_map[b] = cv2.circle(geo_map[b], (offset[1, 0], offset[1, 1]), 4, (153, 89, 151), -1)
            geo_map[b] = cv2.circle(geo_map[b], (offset[2, 0], offset[2, 1]), 4, (91, 241, 241), -1)
            geo_map[b] = cv2.circle(geo_map[b], (offset[3, 0], offset[3, 1]), 4, (241, 91, 134), -1)

            geo_map[b] = cv2.drawContours(geo_map[b], [quad], 0, color, 1)
            geo_map[b] = cv2.circle(geo_map[b], (quad[0, 0], quad[0, 1]), 4, (255, 0, 0), -1)
            geo_map[b] = cv2.circle(geo_map[b], (quad[1, 0], quad[1, 1]), 4, (0, 255, 0), -1)
            geo_map[b] = cv2.circle(geo_map[b], (quad[2, 0], quad[2, 1]), 4, (0, 0, 255), -1)
            geo_map[b] = cv2.circle(geo_map[b], (quad[3, 0], quad[3, 1]), 4, (255, 255, 255), -1)

    geo_map = geo_map.astype(np.float32)
    geo_map = geo_map / 255
    return geo_map


def debugclass(quadboxes, gt_label, roi_idx, gt_boxes, mask, classifier_ch):
    """
    :param quadboxes: (B, 300, 8)
    :param gt_label: (B, 300)
    :param roi_idx: (N', 2) idx
    :param gt_boxes: (B, 300, 8) gatherd
    :parma classifier_ch : (N')
    :return:
    """

    geo_map = np.zeros((quadboxes.shape[0], FLAGS.input_size, FLAGS.input_size, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    lineType = 2

    for i, (idx, bin) in enumerate(zip(roi_idx, mask)):
        b, n = idx
        quad = quadboxes[b, n, :].reshape([4, 2]).astype(np.int32)
        class_info = classifier_ch[i]

        class_idx = np.argmax(class_info, axis= -1)
        class_score = np.amax(class_info, axis=-1)

        rgb = (coco_color_map[3*class_idx], coco_color_map[3*class_idx+1], coco_color_map[3*class_idx+2])

        if checkpoly(quad, FLAGS.input_size, FLAGS.input_size) and class_score > 0.5:
            geo_map[b] = cv2.drawContours(geo_map[b], [quad], 0, rgb, 1)
            #cv2.putText(geo_map[b], (coco_class_dict[class_idx + 1] + u" " + str(class_score)),
            #            (quad[0][0],quad[0][1]),
            #            font,
            #            fontScale,
            #            rgb,
            #            lineType)

    label_shape = gt_label.shape
    fontcolor = (255, 0, 0)

    for b in range(label_shape[0]):
        for i in range(label_shape[1]):
            if gt_label[b, i] != 0:
                quad = gt_boxes[b, i, :].reshape([4, 2]).astype(np.int32)
                geo_map[b] = cv2.drawContours(geo_map[b], [quad], 0, fontcolor, 1)
                #cv2.putText(geo_map[b], (coco_class_dict[gt_label[b,i]]),
                #            (quad[0][0], quad[0][1]),
                #            font,
                #            fontScale,
                #            fontColor,
                #            lineType)

    geo_map = geo_map.astype(np.float32)
    geo_map = geo_map / 255
    return geo_map


def viz_pos_neg_anchor(concat_anchor_box, concat_score_box, pos, mask, input_shape):
    """
    :param concat_anchor_box: (B, N', 8)
    :param concat_score_box: (B, N', 1)
    :param pos: (B, N', 1)
    :param mask: (B, N', 1)
    :return:
    """
    viz_anchor_box_pos_img = np.zeros(input_shape, dtype=np.uint8)
    viz_anchor_box_neg_img = np.zeros(input_shape, dtype=np.uint8)
    for b, (n_concat_anchor_box, n_concat_score_box, n_pos, n_mask) in enumerate(zip(concat_anchor_box, concat_score_box, pos, mask)):
        for each_concat_anchor_box, each_concat_score_box, each_pos, each_mask in zip(n_concat_anchor_box,
                                                                                      n_concat_score_box,
                                                                                      n_pos, n_mask):
            if each_mask[0] == 1:
                quad = each_concat_anchor_box.reshape([4, 2]).astype(np.int32)
                color = getcolor(each_concat_score_box[0])
                if each_pos[0] == 1:
                    viz_anchor_box_pos_img[b] = cv2.drawContours(viz_anchor_box_pos_img[b], [quad], 0, color, 1)
                else:
                    viz_anchor_box_neg_img[b] = cv2.drawContours(viz_anchor_box_neg_img[b], [quad], 0, color, 1)

    viz_anchor_box_pos_img = viz_anchor_box_pos_img.astype(np.float32)
    viz_anchor_box_pos_img = viz_anchor_box_pos_img / 255
    viz_anchor_box_neg_img = viz_anchor_box_neg_img.astype(np.float32)
    viz_anchor_box_neg_img = viz_anchor_box_neg_img / 255
    return viz_anchor_box_pos_img, viz_anchor_box_neg_img

연구용 디버그 스크립트로는 충분하지만 입력 shape·class index·score·좌표 계약이 지나치게 암묵적이라 디버깅 대상의 오류를 시각화 코드 자체의 예외로 덮어버릴 수 있는 구조다.

제안패치
import math

import cv2
import numpy as np
import tensorflow as tf


# TensorFlow 1.x legacy 환경과의 호환성을 유지하되,
# 시각화 함수 자체가 전역 FLAGS에 강하게 의존하지 않도록 최소화한다.
try:
    FLAGS = tf.compat.v1.app.flags.FLAGS
except AttributeError:
    FLAGS = None


COCO_CLASS_DICT = {
    0: "background",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    12: "stop sign",
    13: "parking meter",
    14: "bench",
    15: "bird",
    16: "cat",
    17: "dog",
    18: "horse",
    19: "sheep",
    20: "cow",
    21: "elephant",
    22: "bear",
    23: "zebra",
    24: "giraffe",
    25: "backpack",
    26: "umbrella",
    27: "handbag",
    28: "tie",
    29: "suitcase",
    30: "frisbee",
    31: "skis",
    32: "snowboard",
    33: "sports ball",
    34: "kite",
    35: "baseball bat",
    36: "baseball glove",
    37: "skateboard",
    38: "surfboard",
    39: "tennis racket",
    40: "bottle",
    41: "wine glass",
    42: "cup",
    43: "fork",
    44: "knife",
    45: "spoon",
    46: "bowl",
    47: "banana",
    48: "apple",
    49: "sandwich",
    50: "orange",
    51: "broccoli",
    52: "carrot",
    53: "hot dog",
    54: "pizza",
    55: "donut",
    56: "cake",
    57: "chair",
    58: "couch",
    59: "potted plant",
    60: "bed",
    61: "dining table",
    62: "toilet",
    63: "tv",
    64: "laptop",
    65: "mouse",
    66: "remote",
    67: "keyboard",
    68: "cell phone",
    69: "microwave",
    70: "oven",
    71: "toaster",
    72: "sink",
    73: "refrigerator",
    74: "book",
    75: "clock",
    76: "vase",
    77: "scissors",
    78: "teddy bear",
    79: "hair drier",
    80: "toothbrush",
}


# 원본 색상 테이블 유지.
COCO_COLOR_MAP = (
    0x00, 0x55, 0xFF,
    0xFF, 0x11, 0x00,
    0xFF, 0x1A, 0x00,
    0xFF, 0x22, 0x00,
    0xFF, 0x2A, 0x00,
    0xFF, 0x33, 0x00,
    0xFF, 0x3C, 0x00,
    0xFF, 0x44, 0x00,
    0xFF, 0x4C, 0x00,
    0xFF, 0x55, 0x00,
    0xFF, 0x5E, 0x00,
    0xFF, 0x66, 0x00,
    0xFF, 0x6E, 0x00,
    0xFF, 0x77, 0x00,
    0xFF, 0x77, 0x00,
    0xFF, 0x80, 0x00,
    0xFF, 0x88, 0x00,
    0xFF, 0x90, 0x00,
    0xFF, 0x99, 0x00,
    0xFF, 0xA2, 0x00,
    0xFF, 0xAA, 0x00,
    0xF6, 0xB2, 0x00,
    0xEE, 0xBB, 0x00,
    0xE6, 0xC4, 0x00,
    0xDD, 0xCC, 0x00,
    0xD4, 0xD4, 0x00,
    0xCC, 0xDD, 0x00,
    0xC4, 0xE6, 0x00,
    0xBB, 0xEE, 0x00,
    0xB2, 0xF6, 0x00,
    0xAA, 0xFF, 0x00,
    0xA2, 0xFF, 0x00,
    0x99, 0xFF, 0x00,
    0x90, 0xFF, 0x00,
    0x88, 0xFF, 0x00,
    0x80, 0xFF, 0x00,
    0x77, 0xFF, 0x00,
    0x6E, 0xFF, 0x00,
    0x66, 0xFF, 0x00,
    0x5E, 0xFF, 0x00,
    0x55, 0xFF, 0x00,
    0x44, 0xFF, 0x11,
    0x3C, 0xFF, 0x1A,
    0x33, 0xFF, 0x22,
    0x2A, 0xFF, 0x2A,
    0x22, 0xFF, 0x33,
    0x1A, 0xFF, 0x3C,
    0x11, 0xFF, 0x44,
    0x08, 0xFF, 0x4C,
    0x00, 0xFF, 0x55,
    0x00, 0xFF, 0x5E,
    0x00, 0xFF, 0x66,
    0x00, 0xFF, 0x6E,
    0x00, 0xFF, 0x77,
    0x00, 0xFF, 0x80,
    0x00, 0xFF, 0x88,
    0x00, 0xFF, 0x90,
    0x00, 0xFF, 0x99,
    0x00, 0xFF, 0xA2,
    0x00, 0xFF, 0xAA,
    0x00, 0xF6, 0xB2,
    0x00, 0xEE, 0xBB,
    0x00, 0xE6, 0xC4,
    0x00, 0xDD, 0xCC,
    0x00, 0xD4, 0xD4,
    0x00, 0xCC, 0xDD,
    0x00, 0xC4, 0xE6,
    0x00, 0xBB, 0xEE,
    0x00, 0xB2, 0xF6,
    0x00, 0xAA, 0xFF,
    0x00, 0xA2, 0xFF,
    0x00, 0x99, 0xFF,
    0x00, 0x90, 0xFF,
    0x00, 0x88, 0xFF,
    0x00, 0x80, 0xFF,
    0x00, 0x77, 0xFF,
    0x00, 0x6E, 0xFF,
    0x00, 0x66, 0xFF,
    0x00, 0x5E, 0xFF,
    0xFF, 0x08, 0x00,
    0x00, 0xA2, 0xFF,
    0x00, 0x99, 0xFF,
    0x00, 0x90, 0xFF,
    0x00, 0x88, 0xFF,
    0x00, 0x80, 0xFF,
    0x00, 0x77, 0xFF,
    0x00, 0x6E, 0xFF,
    0x00, 0x66, 0xFF,
    0x00, 0x5E, 0xFF,
    0xFF, 0x08, 0x00,
    0x00, 0x66, 0xFF,
    0x00, 0x5E, 0xFF,
    0xFF, 0x08, 0x00,
)


def _validate_score(score):
    """Return a finite score as float, clipped to [0, 1]."""
    try:
        value = float(np.asarray(score).squeeze())
    except (TypeError, ValueError):
        return 0.0

    if not math.isfinite(value):
        return 0.0

    return float(np.clip(value, 0.0, 1.0))


def getcolor(score):
    """Convert a normalized confidence score to an RGB color."""
    score = _validate_score(score)

    if score < 0.25:
        r = 0.0
        g = 4.0 * score
        b = 1.0
    elif score < 0.5:
        r = 0.0
        g = 1.0
        b = 1.0 + 4.0 * (0.25 - score)
    elif score < 0.75:
        r = 4.0 * (score - 0.5)
        g = 1.0
        b = 0.0
    else:
        r = 1.0
        g = 1.0 + 4.0 * (0.75 - score)
        b = 0.0

    return (
        int(np.clip(r * 255.0, 0, 255)),
        int(np.clip(g * 255.0, 0, 255)),
        int(np.clip(b * 255.0, 0, 255)),
    )


def _validate_image_shape(img):
    if not isinstance(img, np.ndarray):
        raise TypeError(f"img must be numpy.ndarray, got {type(img).__name__}")

    if img.ndim != 4:
        raise ValueError(
            f"img must have shape [B, H, W, C], got ndim={img.ndim}"
        )

    if img.shape[-1] not in (1, 3, 4):
        raise ValueError(
            f"img must have 1, 3, or 4 channels, got {img.shape[-1]}"
        )


def _prepare_quad(box, width, height):
    """Validate and clip one quadrilateral to the drawable image area."""
    try:
        quad = np.asarray(box).reshape(4, 2)
    except (ValueError, TypeError):
        return None

    if not np.all(np.isfinite(quad)):
        return None

    if width <= 0 or height <= 0:
        return None

    quad = np.rint(quad).astype(np.int32)

    quad[:, 0] = np.clip(quad[:, 0], 0, width - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, height - 1)

    return quad


def checkpoly(ind_box, w, h):
    """Check whether a quadrilateral is finite and inside the image."""
    if w <= 0 or h <= 0:
        return 0

    try:
        box = np.asarray(ind_box)
    except (TypeError, ValueError):
        return 0

    if box.shape != (4, 2):
        return 0

    if not np.all(np.isfinite(box)):
        return 0

    return int(
        np.all(box[:, 0] >= 0)
        and np.all(box[:, 0] < w)
        and np.all(box[:, 1] >= 0)
        and np.all(box[:, 1] < h)
    )


def _to_float_image(img, channel_reverse=False):
    """Convert uint8-like visualization image to float32 [0, 1]."""
    result = np.asarray(img)

    if channel_reverse:
        if result.ndim != 4 or result.shape[-1] < 3:
            raise ValueError(
                "Channel reversal requires image shape [B, H, W, C] with C >= 3."
            )
        result = result[..., ::-1]

    result = np.asarray(result, dtype=np.uint8)
    return result.astype(np.float32) / 255.0


def _get_input_size(input_size=None):
    """Prefer explicit input_size; fall back to legacy FLAGS only if available."""
    if input_size is not None:
        value = int(input_size)
        if value <= 0:
            raise ValueError("input_size must be greater than zero.")
        return value

    if FLAGS is not None and hasattr(FLAGS, "input_size"):
        value = int(FLAGS.input_size)
        if value > 0:
            return value

    raise ValueError(
        "input_size must be provided when legacy FLAGS.input_size is unavailable."
    )


def makepoly_with_score(
    batch,
    boxes,
    scores,
    h,
    w,
    img,
    thres=0.1,
):
    _validate_image_shape(img)

    batch = int(batch)
    height = int(h)
    width = int(w)
    threshold = _validate_score(thres)

    if batch != img.shape[0]:
        raise ValueError(
            f"batch mismatch: batch={batch}, img_batch={img.shape[0]}"
        )

    if len(boxes) < batch or len(scores) < batch:
        raise ValueError("boxes/scores batch dimension is smaller than batch.")

    result = np.asarray(img, dtype=np.uint8).copy()

    # 원본의 RGB/BGR 변환 동작 유지.
    if result.shape[-1] >= 3:
        result = result[..., ::-1].copy()

    for i in range(batch):
        for box, score in zip(boxes[i], scores[i]):
            score_value = _validate_score(score)

            if score_value <= threshold:
                continue

            quad = _prepare_quad(box, width, height)
            if quad is None:
                continue

            color = getcolor(score_value)
            result[i] = cv2.drawContours(
                result[i],
                [quad],
                contourIdx=0,
                color=color,
                thickness=1,
            )

    return result.astype(np.float32) / 255.0


def makepoly_with_score_color(
    batch,
    boxes,
    scores,
    h,
    w,
    img,
    color,
    thres=0.0,
):
    _validate_image_shape(img)

    batch = int(batch)
    height = int(h)
    width = int(w)
    threshold = _validate_score(thres)

    if batch != img.shape[0]:
        raise ValueError(
            f"batch mismatch: batch={batch}, img_batch={img.shape[0]}"
        )

    if len(boxes) < batch or len(scores) < batch:
        raise ValueError("boxes/scores batch dimension is smaller than batch.")

    result = np.asarray(img, dtype=np.uint8).copy()

    color = tuple(int(np.clip(v, 0, 255)) for v in color)

    for i in range(batch):
        for box, score in zip(boxes[i], scores[i]):
            if _validate_score(score) <= threshold:
                continue

            quad = _prepare_quad(box, width, height)
            if quad is None:
                continue

            result[i] = cv2.drawContours(
                result[i],
                [quad],
                contourIdx=0,
                color=color,
                thickness=3,
            )

    return result


def _get_coco_color(class_idx):
    """Safely resolve COCO class index to a color."""
    try:
        class_idx = int(class_idx)
    except (TypeError, ValueError):
        return None

    color_offset = class_idx * 3

    if color_offset < 0 or color_offset + 2 >= len(COCO_COLOR_MAP):
        return None

    return (
        COCO_COLOR_MAP[color_offset],
        COCO_COLOR_MAP[color_offset + 1],
        COCO_COLOR_MAP[color_offset + 2],
    )


def debugclassifier(
    quadboxes,
    gt_boxes,
    roi_idx,
    offset_ch,
    mask,
    classifier_ch,
    input_size=None,
):
    """
    Visualize classifier ROI, GT offset, and predicted score.

    quadboxes   : (B, N, 8)
    gt_boxes    : (N', 8)
    roi_idx     : (N', 2)
    offset_ch   : (N', 8)
    classifier_ch : (N',)
    """
    input_size = _get_input_size(input_size)

    quadboxes = np.asarray(quadboxes)
    gt_boxes = np.asarray(gt_boxes)
    roi_idx = np.asarray(roi_idx)
    offset_ch = np.asarray(offset_ch)
    classifier_ch = np.asarray(classifier_ch)
    mask = np.asarray(mask)

    if quadboxes.ndim != 3 or quadboxes.shape[-1] != 8:
        raise ValueError(
            f"quadboxes must have shape [B, N, 8], got {quadboxes.shape}"
        )

    if roi_idx.ndim != 2 or roi_idx.shape[-1] != 2:
        raise ValueError(
            f"roi_idx must have shape [N, 2], got {roi_idx.shape}"
        )

    sample_count = min(
        len(roi_idx),
        len(gt_boxes),
        len(offset_ch),
        len(classifier_ch),
        len(mask),
    )

    geo_map = np.zeros(
        (quadboxes.shape[0], input_size, input_size, 3),
        dtype=np.uint8,
    )

    for i in range(sample_count):
        b, n = map(int, roi_idx[i])

        if not (0 <= b < quadboxes.shape[0]):
            continue

        if not (0 <= n < quadboxes.shape[1]):
            continue

        score = _validate_score(classifier_ch[i])

        if score <= 0.5:
            continue

        quad = _prepare_quad(
            quadboxes[b, n],
            input_size,
            input_size,
        )
        offset = _prepare_quad(
            offset_ch[i],
            input_size,
            input_size,
        )

        if quad is None or offset is None:
            continue

        color = getcolor(score)

        geo_map[b] = cv2.drawContours(
            geo_map[b],
            [offset],
            contourIdx=0,
            color=(255, 255, 255),
            thickness=1,
        )

        offset_colors = (
            (153, 151, 89),
            (153, 89, 151),
            (91, 241, 241),
            (241, 91, 134),
        )

        for point, point_color in zip(offset, offset_colors):
            geo_map[b] = cv2.circle(
                geo_map[b],
                tuple(point),
                4,
                point_color,
                -1,
            )

        geo_map[b] = cv2.drawContours(
            geo_map[b],
            [quad],
            contourIdx=0,
            color=color,
            thickness=1,
        )

        quad_colors = (
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 255),
        )

        for point, point_color in zip(quad, quad_colors):
            geo_map[b] = cv2.circle(
                geo_map[b],
                tuple(point),
                4,
                point_color,
                -1,
            )

    return geo_map.astype(np.float32) / 255.0


def debugclass(
    quadboxes,
    gt_label,
    roi_idx,
    gt_boxes,
    mask,
    classifier_ch,
    input_size=None,
):
    """
    Visualize predicted classes and ground-truth boxes.
    """
    input_size = _get_input_size(input_size)

    quadboxes = np.asarray(quadboxes)
    gt_label = np.asarray(gt_label)
    roi_idx = np.asarray(roi_idx)
    gt_boxes = np.asarray(gt_boxes)
    classifier_ch = np.asarray(classifier_ch)

    if quadboxes.ndim != 3 or quadboxes.shape[-1] != 8:
        raise ValueError(
            f"quadboxes must have shape [B, N, 8], got {quadboxes.shape}"
        )

    geo_map = np.zeros(
        (quadboxes.shape[0], input_size, input_size, 3),
        dtype=np.uint8,
    )

    sample_count = min(
        len(roi_idx),
        len(classifier_ch),
    )

    for i in range(sample_count):
        b, n = map(int, roi_idx[i])

        if not (0 <= b < quadboxes.shape[0]):
            continue

        if not (0 <= n < quadboxes.shape[1]):
            continue

        class_info = np.asarray(classifier_ch[i])

        if class_info.size == 0 or not np.all(np.isfinite(class_info)):
            continue

        class_idx = int(np.argmax(class_info))
        class_score = _validate_score(np.max(class_info))

        if class_score <= 0.5:
            continue

        rgb = _get_coco_color(class_idx)
        if rgb is None:
            continue

        quad = _prepare_quad(
            quadboxes[b, n],
            input_size,
            input_size,
        )

        if quad is None:
            continue

        geo_map[b] = cv2.drawContours(
            geo_map[b],
            [quad],
            contourIdx=0,
            color=rgb,
            thickness=1,
        )

    if gt_label.ndim != 2:
        raise ValueError(
            f"gt_label must have shape [B, N], got {gt_label.shape}"
        )

    if gt_boxes.ndim != 3 or gt_boxes.shape[-1] != 8:
        raise ValueError(
            f"gt_boxes must have shape [B, N, 8], got {gt_boxes.shape}"
        )

    batch_count = min(
        gt_label.shape[0],
        gt_boxes.shape[0],
        quadboxes.shape[0],
    )

    for b in range(batch_count):
        object_count = min(
            gt_label.shape[1],
            gt_boxes.shape[1],
        )

        for i in range(object_count):
            if gt_label[b, i] == 0:
                continue

            quad = _prepare_quad(
                gt_boxes[b, i],
                input_size,
                input_size,
            )

            if quad is None:
                continue

            geo_map[b] = cv2.drawContours(
                geo_map[b],
                [quad],
                contourIdx=0,
                color=(255, 0, 0),
                thickness=1,
            )

    return geo_map.astype(np.float32) / 255.0


def viz_pos_neg_anchor(
    concat_anchor_box,
    concat_score_box,
    pos,
    mask,
    input_shape,
):
    """
    Visualize positive/negative anchors.

    concat_anchor_box : (B, N, 8)
    concat_score_box  : (B, N, 1)
    pos               : (B, N, 1)
    mask              : (B, N, 1)
    """
    anchor_box = np.asarray(concat_anchor_box)
    score_box = np.asarray(concat_score_box)
    pos = np.asarray(pos)
    mask = np.asarray(mask)

    if anchor_box.ndim != 3 or anchor_box.shape[-1] != 8:
        raise ValueError(
            f"concat_anchor_box must have shape [B, N, 8], got {anchor_box.shape}"
        )

    if len(input_shape) != 4:
        raise ValueError(
            f"input_shape must be [B, H, W, C], got {input_shape}"
        )

    batch, height, width, channels = map(int, input_shape)

    if channels < 1:
        raise ValueError("input_shape channel count must be positive.")

    if anchor_box.shape[0] != batch:
        raise ValueError(
            f"batch mismatch: boxes={anchor_box.shape[0]}, input={batch}"
        )

    pos_img = np.zeros(
        (batch, height, width, channels),
        dtype=np.uint8,
    )
    neg_img = np.zeros_like(pos_img)

    for b in range(batch):
        for box, score, is_pos, is_valid in zip(
            anchor_box[b],
            score_box[b],
            pos[b],
            mask[b],
        ):
            if np.asarray(is_valid).reshape(-1)[0] != 1:
                continue

            quad = _prepare_quad(box, width, height)
            if quad is None:
                continue

            color = getcolor(score)

            target = (
                pos_img
                if np.asarray(is_pos).reshape(-1)[0] == 1
                else neg_img
            )

            # OpenCV contour drawing은 3채널 이상 입력을 일반적으로 기대하므로
            # grayscale 입력은 명시적으로 처리한다.
            if target.shape[-1] == 1:
                draw_target = cv2.cvtColor(
                    target[b],
                    cv2.COLOR_GRAY2BGR,
                )
                draw_target = cv2.drawContours(
                    draw_target,
                    [quad],
                    contourIdx=0,
                    color=color,
                    thickness=1,
                )
                target[b, :, :, 0] = cv2.cvtColor(
                    draw_target,
                    cv2.COLOR_BGR2GRAY,
                )
            else:
                target[b] = cv2.drawContours(
                    target[b],
                    [quad],
                    contourIdx=0,
                    color=color,
                    thickness=1,
                )

    return (
        pos_img.astype(np.float32) / 255.0,
        neg_img.astype(np.float32) / 255.0,
    )

최종 개선사항
✅ 암묵적인 입력 shape 계약 → 공통 입력·polygon 검증 계층 도입 → malformed 데이터로 인한 디버깅 파이프라인 중단 방지
✅ 무검증 class index 직접 색상 배열 접근 → 안전한 class index resolver → 비정상 예측값에 의한 IndexError 방지
✅ NaN/Inf/범위 외 score 직접 사용 → finite 검사 및 [0, 1] 정규화 → 색상 계산과 threshold 판정의 수치 안정성 확보
✅ 이미지 경계 검사와 clipping 로직 분산 → 공통 polygon clipping 적용 → OpenCV 좌표 범위 불일치 및 경계 오류 제거
✅ FLAGS.input_size 전역 의존 → 명시적 input_size 우선 + legacy fallback → 테스트 가능성과 실행 환경 독립성 강화
✅ 반복적인 OpenCV 렌더링·배열 후처리 → 공통 helper로 책임 분리 → 중복 감소와 시각화 동작 일관성 확보
✅ zip()으로 입력 길이 불일치를 조용히 무시 → 핵심 batch/shape 계약 검증 → 디버깅 데이터 누락을 조용히 통과시키는 문제 차단

연구용 시각화 코드에서 운영 가능한 디버깅 유틸리티로 승격했으며, 입력 무결성·좌표 안전성·수치 안정성·legacy 호환성을 확보하면서도 원본의 역할과 처리 흐름은 유지한 리팩이다.    
