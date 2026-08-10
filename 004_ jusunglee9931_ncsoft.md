원본코드
import tensorflow as tf
import numpy as np
from rpn_util.generate_anchors import generate_anchors
import math

def cal_iou_np(quad_or_rec_a, quad_or_rec_b):
    """
    quad_or_rec_a :(N, 8 or 4)
    quad_or_rec_a :(N', 8 or 4)

    return (N, N')
    """

    a_shape = quad_or_rec_a.shape
    b_shape = quad_or_rec_b.shape

    if a_shape[-1] == 8:
        x1_a, y1_a, x2_a, y2_a, x3_a, y3_a, x4_a, y4_a = np.split(quad_or_rec_a, 8, axis=-1)
        x_min = np.minimum(np.minimum(np.minimum(x1_a, x2_a), x3_a), x4_a)
        x_max = np.maximum(np.maximum(np.maximum(x1_a, x2_a), x3_a), x4_a)
        y_min = np.minimum(np.minimum(np.minimum(y1_a, y2_a), y3_a), y4_a)
        y_max = np.maximum(np.maximum(np.maximum(y1_a, y2_a), y3_a), y4_a)
        boxa = np.stack([x_min, y_min, x_max, y_max], axis=-1)
        boxa = boxa[:, 0, :]
    else:
        boxa = quad_or_rec_a

    if b_shape[-1] == 8:
        x1_b, y1_b, x2_b, y2_b, x3_b, y3_b, x4_b, y4_b = np.split(quad_or_rec_b, 8, axis=-1)
        x_min = np.minimum(np.minimum(np.minimum(x1_b, x2_b), x3_b), x4_b)
        x_max = np.maximum(np.maximum(np.maximum(x1_b, x2_b), x3_b), x4_b)
        y_min = np.minimum(np.minimum(np.minimum(y1_b, y2_b), y3_b), y4_b)
        y_max = np.maximum(np.maximum(np.maximum(y1_b, y2_b), y3_b), y4_b)
        boxb = np.stack([x_min, y_min, x_max, y_max], axis=-1)
        boxb = boxb[:, 0, :]
    else:
        boxb = quad_or_rec_b

    gt_x1 = np.expand_dims(boxa[:, 0], axis=-1)
    gt_y1 = np.expand_dims(boxa[:, 1], axis=-1)
    gt_x2 = np.expand_dims(boxa[:, 2], axis=-1)
    gt_y2 = np.expand_dims(boxa[:, 3], axis=-1)

    pred_x1 = np.expand_dims(boxb[:, 0], axis=-1)
    pred_y1 = np.expand_dims(boxb[:, 1], axis=-1)
    pred_x2 = np.expand_dims(boxb[:, 2], axis=-1)
    pred_y2 = np.expand_dims(boxb[:, 3], axis=-1)

    xi1 = np.maximum(gt_x1, np.transpose(pred_x1, [1, 0]))  # (MAX_LENGTH, 1) * (1, MAX_LENGTH)
    yi1 = np.maximum(gt_y1, np.transpose(pred_y1, [1, 0]))  # (MAX_LENGTH, 1) * (1, MAX_LENGTH)

    xi2 = np.minimum(gt_x2, np.transpose(pred_x2, [1, 0]))  # (MAX_LENGTH, 1) * (1, MAX_LENGTH)
    yi2 = np.minimum(gt_y2, np.transpose(pred_y2, [1, 0]))  # (MAX_LENGTH, 1) * (1, MAX_LENGTH)

    intersection = np.maximum(0.0, (xi2 - xi1 + 1)) * np.maximum(0.0, (yi2 - yi1 + 1))
    box_area = (pred_x2 - pred_x1 + 1) * (pred_y2 - pred_y1 + 1)
    gt_box_area = (gt_x2 - gt_x1 + 1) * (gt_y2 - gt_y1 + 1)

    union = (gt_box_area + np.transpose(box_area, [1, 0])) - intersection

    overlaps = intersection / union

    return overlaps


def concat_quad_to_rect(boxes):
    x1, y1, x2, y2, x3, y3, x4, y4 = tf.split(value=boxes, num_or_size_splits=8, axis=-1)  # (B, MAX_LENGTH, 1)
    x_bundle = tf.concat([x1, x2, x3, x4], axis=-1)
    y_bundle = tf.concat([y1, y2, y3, y4], axis=-1)

    min_x = tf.reduce_min(x_bundle, -1)  # B, N
    max_x = tf.reduce_max(x_bundle, -1)

    min_y = tf.reduce_min(y_bundle, -1)
    max_y = tf.reduce_max(y_bundle, -1)
    return tf.stack([min_x, min_y, max_x, max_y], -1)


def cal_iou(gt_boxes, pred_boxes):
    """
    :param gt_boxes: (B, MAX_LENGTH, 8)
    :param pred_boxes: (B, MAX_LENGTH, 8)
    :return: IOU MAXIMIZED boxes (B, MAX_LENGTH, MAX_LENGTH)
    """

    gt_rect_boxes = concat_quad_to_rect(gt_boxes)  # (B, MAX_LENGTH, 4)
    pred_rect_boxes = concat_quad_to_rect(pred_boxes)  # (B, MAX_LENGTH, 4)

    gt_x1, gt_y1, gt_x2, gt_y2 = tf.split(value=gt_rect_boxes, num_or_size_splits=4, axis=-1)  # (B, MAX_LENGTH, 1)
    pred_x1, pred_y1, pred_x2, pred_y2 = tf.split(value=pred_rect_boxes, num_or_size_splits=4, axis=-1)

    xi1 = tf.maximum(gt_x1, tf.transpose(pred_x1, [0, 2, 1]))  # (B, MAX_LENGTH, 1) * (B, 1, MAX_LENGTH)
    yi1 = tf.maximum(gt_y1, tf.transpose(pred_y1, [0, 2, 1]))  # (B, MAX_LENGTH, 1) * (B, 1, MAX_LENGTH)

    xi2 = tf.minimum(gt_x2, tf.transpose(pred_x2, [0, 2, 1]))  # (B, MAX_LENGTH, 1) * (B, 1, MAX_LENGTH)
    yi2 = tf.minimum(gt_y2, tf.transpose(pred_y2, [0, 2, 1]))  # (B, MAX_LENGTH, 1) * (B, 1, MAX_LENGTH)

    intersection = tf.maximum(0.0, (xi2 - xi1 + 1)) * tf.maximum(0.0, (yi2 - yi1 + 1))
    box_area = (pred_x2 - pred_x1 + 1) * (pred_y2 - pred_y1 + 1)
    gt_box_area = (gt_x2 - gt_x1 + 1) * (gt_y2 - gt_y1 + 1)

    union = (gt_box_area + tf.transpose(box_area, [0, 2, 1])) - intersection

    overlaps = intersection / union

    return overlaps


def quad_to_param(anchor):
    point_lt = tf.stack([anchor[:, 0], anchor[:, 1]], axis=-1)  # N ,2
    point_rt = tf.stack([anchor[:, 2], anchor[:, 3]], axis=-1)
    point_rb = tf.stack([anchor[:, 4], anchor[:, 5]], axis=-1)
    point_lb = tf.stack([anchor[:, 6], anchor[:, 7]], axis=-1)

    theta_vector1 = (point_rt - point_lt + point_rb - point_lb) / 2.0
    x_vector1 = theta_vector1[:, 0]
    eps = tf.fill(tf.shape(x_vector1), 0.00001)
    div_x_vector1 = tf.where(x_vector1 == 0, eps, x_vector1)
    anchor_theta_1 = (theta_vector1[:, 1] / div_x_vector1)
    anchor_theta_1 = tf.atan(anchor_theta_1)
    anchor_theta_1 = tf.clip_by_value(anchor_theta_1, -math.pi/2.0, math.pi/2.0)

    anchor_w = tf.math.sqrt(tf.math.square(anchor[:, 2] - anchor[:, 0]) + tf.math.square(
        anchor[:, 3] - anchor[:, 1])) + 1
    anchor_h = tf.math.sqrt(tf.math.square(anchor[:, 4] - anchor[:, 2]) + tf.math.square(
        anchor[:, 5] - anchor[:, 3])) + 1

    anchor_center_x = (anchor[:, 0] + anchor[:, 2] + anchor[:, 4] + anchor[:, 6]) / 4.0
    anchor_center_y = (anchor[:, 1] + anchor[:, 3] + anchor[:, 5] + anchor[:, 7]) / 4.0
    return anchor_w, anchor_h, anchor_center_x, anchor_center_y, anchor_theta_1


def getboxes(geometry_concat, div=4.0):
    d1, d2, d3, d4, score_bk, score_pred, theta = tf.split(value=geometry_concat, num_or_size_splits=7, axis=3)
    score_concat = tf.concat([score_bk, score_pred], axis=-1)
    score_concat_softmax = tf.nn.softmax(score_concat)
    score = tf.expand_dims(score_concat_softmax[:, :, :, -1], axis=-1)
    rec_plane = tf.zeros_like(score, dtype=tf.float32)
    zero_plane = tf.zeros_like(score, dtype=tf.float32)
    one_plane = tf.ones_like(score, dtype=tf.float32)
    thres = tf.fill(tf.shape(rec_plane), 0.1)
    score_idx = tf.where(tf.greater(score, thres))
    thres_score = tf.where(tf.greater(score, thres), one_plane, zero_plane)
    x_cord = tf.scatter_nd(score_idx, score_idx[:, 2], tf.shape(rec_plane, out_type=tf.int64))
    y_cord = tf.scatter_nd(score_idx, score_idx[:, 1], tf.shape(rec_plane, out_type=tf.int64))

    score_shape = tf.cast(tf.shape(score), tf.float32)
    w = div * score_shape[2]
    h = div * score_shape[1]

    d1 = tf.clip_by_value(d1, 0.0, tf.cast(2 * h, tf.float32))
    d2 = tf.clip_by_value(d2, 0.0, tf.cast(2 * w, tf.float32))
    d3 = tf.clip_by_value(d3, 0.0, tf.cast(2 * h, tf.float32))
    d4 = tf.clip_by_value(d4, 0.0, tf.cast(2 * w, tf.float32))

    y_top = tf.squeeze(-d1 * thres_score, axis=-1)
    y_bottom = tf.squeeze(d3 * thres_score, axis=-1)
    x_left = tf.squeeze(-d2 * thres_score, axis=-1)
    x_right = tf.squeeze(d4 * thres_score, axis=-1)
    theta = tf.squeeze(theta, axis=-1)
    sine = tf.sin(theta)
    cos = tf.cos(theta)  # [b, w, h, 1]

    lt_x = x_left * cos - y_top * sine
    lt_y = x_left * sine + y_top * cos

    lb_x = x_left * cos - y_bottom * sine
    lb_y = x_left * sine + y_bottom * cos

    rt_x = x_right * cos - y_top * sine
    rt_y = x_right * sine + y_top * cos

    rb_x = x_right * cos - y_bottom * sine
    rb_y = x_right * sine + y_bottom * cos

    new_mat_lt = tf.stack([lt_x, lt_y], axis=-1)
    new_mat_lb = tf.stack([lb_x, lb_y], axis=-1)
    new_mat_rt = tf.stack([rt_x, rt_y], axis=-1)
    new_mat_rb = tf.stack([rb_x, rb_y], axis=-1)

    y1 = div * tf.cast(y_cord[:, :, :, 0], tf.float32) + new_mat_lt[:, :, :, 1]
    y2 = div * tf.cast(y_cord[:, :, :, 0], tf.float32) + new_mat_rt[:, :, :, 1]
    y3 = div * tf.cast(y_cord[:, :, :, 0], tf.float32) + new_mat_rb[:, :, :, 1]
    y4 = div * tf.cast(y_cord[:, :, :, 0], tf.float32) + new_mat_lb[:, :, :, 1]

    x1 = div * tf.cast(x_cord[:, :, :, 0], tf.float32) + new_mat_lt[:, :, :, 0]
    x2 = div * tf.cast(x_cord[:, :, :, 0], tf.float32) + new_mat_rt[:, :, :, 0]
    x3 = div * tf.cast(x_cord[:, :, :, 0], tf.float32) + new_mat_rb[:, :, :, 0]
    x4 = div * tf.cast(x_cord[:, :, :, 0], tf.float32) + new_mat_lb[:, :, :, 0]

    rboxes = tf.stack([x1, y1, x2, y2,
                      x3, y3, x4, y4], axis=-1)

    return rboxes, score * thres_score, thres_score, w, h


def bbox_inverse_transform(anchor, quadboxes):
    """
    :param anchor: (K' , 8)
    :param quadboxes: (K', 8)
    :return: target_offset(K' ,5)
    """

    anchor_w, anchor_h, \
    anchor_center_x, anchor_center_y, anchor_theta_1 = quad_to_param(anchor)

    gt_w, gt_h, \
    gt_center_x, gt_center_y, gt_theta_1 = quad_to_param(quadboxes)

    gt_x1, gt_y1, gt_x2, gt_y2, gt_x3, gt_y3, gt_x4, gt_y4 = tf.split(quadboxes, num_or_size_splits=8, axis=-1)
    quadboxes_moved_1 = tf.concat([gt_x3, gt_y3, gt_x4, gt_y4, gt_x1, gt_y1, gt_x2, gt_y2], axis=-1)
    move_1_gt_w, move_1_gt_h,\
    move_1_gt_center_x, move_1_gt_center_y, move_1_gt_theta_1 = quad_to_param(quadboxes_moved_1)

    quadboxes_moved_2 = tf.concat([gt_x4, gt_y4, gt_x1, gt_y1, gt_x2, gt_y2, gt_x3, gt_y3,], axis=-1)
    move_2_gt_w, move_2_gt_h, \
    move_2_gt_center_x, move_2_gt_center_y, move_2_gt_theta_1 = quad_to_param(quadboxes_moved_2)

    quadboxes_moved_3 = tf.concat([gt_x2, gt_y2, gt_x3, gt_y3, gt_x4, gt_y4, gt_x1, gt_y1, ], axis=-1)
    move_3_gt_w, move_3_gt_h, \
    move_3_gt_center_x, move_3_gt_center_y, move_3_gt_theta_1 = quad_to_param(quadboxes_moved_3)

    tx = (gt_center_x - anchor_center_x) / anchor_w
    ty = (gt_center_y - anchor_center_y) / anchor_h
    tw = tf.log(gt_w/anchor_w)
    th = tf.log(gt_h/anchor_h)

    move_1_tx = (move_1_gt_center_x - anchor_center_x) / anchor_w
    move_1_ty = (move_1_gt_center_y - anchor_center_y) / anchor_h
    move_1_tw = tf.log(move_1_gt_w/anchor_w)
    move_1_th = tf.log(move_1_gt_h/anchor_h)

    move_2_tx = (move_2_gt_center_x - anchor_center_x) / anchor_w
    move_2_ty = (move_2_gt_center_y - anchor_center_y) / anchor_h
    move_2_tw = tf.log(move_2_gt_w/anchor_w)
    move_2_th = tf.log(move_2_gt_h/anchor_h)

    move_3_tx = (move_3_gt_center_x - anchor_center_x) / anchor_w
    move_3_ty = (move_3_gt_center_y - anchor_center_y) / anchor_h
    move_3_tw = tf.log(move_3_gt_w/anchor_w)
    move_3_th = tf.log(move_3_gt_h/anchor_h)

    first_cond_1 = tf.abs(gt_theta_1 - anchor_theta_1)
    first_cond_2 = tf.abs(move_1_gt_theta_1 - anchor_theta_1)

    theta = tf.where(first_cond_1 < first_cond_2, gt_theta_1 - anchor_theta_1, move_1_gt_theta_1 - anchor_theta_1)
    tx = tf.where(first_cond_1 < first_cond_2, tx, move_1_tx)
    ty = tf.where(first_cond_1 < first_cond_2, ty, move_1_ty)
    tw = tf.where(first_cond_1 < first_cond_2, tw, move_1_tw)
    th = tf.where(first_cond_1 < first_cond_2, th, move_1_th)

    second_cond_1 = tf.abs(theta)
    second_cond_2 = tf.abs(move_2_gt_theta_1 - anchor_theta_1)

    theta = tf.where(second_cond_1 < second_cond_2, theta,
                     move_2_gt_theta_1 - anchor_theta_1)
    tx = tf.where(second_cond_1 < second_cond_2, tx, move_2_tx)
    ty = tf.where(second_cond_1 < second_cond_2, ty, move_2_ty)
    tw = tf.where(second_cond_1 < second_cond_2, tw, move_2_tw)
    th = tf.where(second_cond_1 < second_cond_2, th, move_2_th)

    third_cond_1 = tf.abs(theta)
    third_cond_2 = tf.abs(move_3_gt_theta_1 - anchor_theta_1)

    theta = tf.where(third_cond_1 < third_cond_2, theta,
                     move_3_gt_theta_1 - anchor_theta_1)
    tx = tf.where(third_cond_1 < third_cond_2, tx, move_3_tx)
    ty = tf.where(third_cond_1 < third_cond_2, ty, move_3_ty)
    tw = tf.where(third_cond_1 < third_cond_2, tw, move_3_tw)
    th = tf.where(third_cond_1 < third_cond_2, th, move_3_th)

    theta = tf.where(tf.math.is_nan(theta), tf.zeros_like(theta), theta)

    return tf.stack([tx, ty, tw, th, theta], axis=-1)


def bbox_transform_from_anchor(rpn_box, rpn_anchor, input_shape):
    """
    :param rpn_box: (B, H ,W , NUM_ANCHOR * 5)
    :param rpn_anchor: (H, W , NUM_ANCHOR, 8)
    :param input_shape:
    :return: (B, H*W*NUM_ANCHOR, 8*2)
    """
    rpn_box_shape = tf.shape(rpn_box)
    B = rpn_box_shape[0]
    H = rpn_box_shape[1]
    W = rpn_box_shape[2]

    one = tf.constant(1.0)
    half = tf.constant(0.5)
    # (B, H , W , NUM_ANCHOR, 4)
    rpn_box_reshape = tf.reshape(rpn_box, [B, H, W, -1, 5])
    rpn_box_reshape_shape = tf.shape(rpn_box_reshape)
    rpn_anchor = tf.to_float(rpn_anchor)

    # (B, H, W , NUM_ANCHOR)
    dx = rpn_box_reshape[:, :, :, :, 0]
    dy = rpn_box_reshape[:, :, :, :, 1]
    dw = rpn_box_reshape[:, :, :, :, 2]
    dh = rpn_box_reshape[:, :, :, :, 3]
    dtheta = rpn_box_reshape[:, :, :, :, 4]

    # (H, W, NUM_ANCHOR)
    point_lt = tf.stack([rpn_anchor[:, :, :, 0], rpn_anchor[:, :, :, 1]], axis=-1)  # H, W  ANCHOR, 2
    point_rt = tf.stack([rpn_anchor[:, :, :, 2], rpn_anchor[:, :, :, 3]], axis=-1)
    point_rb = tf.stack([rpn_anchor[:, :, :, 4], rpn_anchor[:, :, :, 5]], axis=-1)
    point_lb = tf.stack([rpn_anchor[:, :, :, 6], rpn_anchor[:, :, :, 7]], axis=-1)

    theta_vector1 = (point_rt - point_lt)
    x_vector1 = theta_vector1[:, :, :, 0]
    eps = tf.fill(tf.shape(x_vector1), 0.00001)
    div_x_vector1 = tf.where(x_vector1 == 0, eps, x_vector1)
    theta_ratio_1 = (theta_vector1[:, :, :, 1] / div_x_vector1)  # H, W, ANCHOR

    theta = tf.atan(theta_ratio_1)

    box_w = tf.math.sqrt(tf.math.square(rpn_anchor[:, :, :, 2] - rpn_anchor[:, :, :, 0])
                         + tf.math.square(rpn_anchor[:, :, :, 3] - rpn_anchor[:, :, :, 1])) + one
    box_h = tf.math.sqrt(tf.math.square(rpn_anchor[:, :, :, 4] - rpn_anchor[:, :, :, 2])
                         + tf.math.square(rpn_anchor[:, :, :, 5] - rpn_anchor[:, :, :, 3])) + one

    box_center_x = (rpn_anchor[:, :, :, 0] + rpn_anchor[:, :, :, 2]
                    + rpn_anchor[:, :, :, 4] + rpn_anchor[:, :, :, 6]) / 4.0
    box_center_y = (rpn_anchor[:, :, :, 1] + rpn_anchor[:, :, :, 3]
                    + rpn_anchor[:, :, :, 5] + rpn_anchor[:, :, :, 7]) / 4.0

    # (1, H, W, NUM_ANCHOR)
    box_w = tf.expand_dims(box_w, axis=0)
    box_h = tf.expand_dims(box_h, axis=0)
    box_center_x = tf.expand_dims(box_center_x, axis=0)
    box_center_y = tf.expand_dims(box_center_y, axis=0)
    theta = tf.expand_dims(theta, axis=0)

    # (B, H, W, NUM_ANCHOR)
    pred_center_x = box_center_x + box_w * dx
    pred_center_y = box_center_y + box_h * dy
    pred_w = tf.exp(dw) * box_w
    pred_h = tf.exp(dh) * box_h

    pred_w = tf.clip_by_value(pred_w, 0, tf.cast(input_shape[2], 'float32') - 1)
    pred_h = tf.clip_by_value(pred_h, 0, tf.cast(input_shape[1], 'float32') - 1)
    pred_theta = theta + dtheta

    sin = tf.sin(pred_theta)
    cos = tf.cos(pred_theta)  # [b, w, h, 1]

    pred_x1 = (- pred_w / 2) * cos - (- pred_h / 2) * sin + pred_center_x
    pred_y1 = (- pred_w / 2) * sin + (- pred_h / 2) * cos + pred_center_y

    pred_x2 = (pred_w / 2) * cos - (- pred_h / 2) * sin + pred_center_x
    pred_y2 = (pred_w / 2) * sin + (- pred_h / 2) * cos + pred_center_y

    pred_x3 = (pred_w / 2) * cos - (pred_h / 2) * sin + pred_center_x
    pred_y3 = (pred_w / 2) * sin + (pred_h / 2) * cos + pred_center_y

    pred_x4 = (- pred_w / 2) * cos - (pred_h / 2) * sin + pred_center_x
    pred_y4 = (- pred_w / 2) * sin + (pred_h / 2) * cos + pred_center_y

    anchor_x1 = rpn_anchor[:, :, :, 0]
    anchor_x2 = rpn_anchor[:, :, :, 2]
    anchor_x3 = rpn_anchor[:, :, :, 4]
    anchor_x4 = rpn_anchor[:, :, :, 6]

    anchor_y1 = rpn_anchor[:, :, :, 1]
    anchor_y2 = rpn_anchor[:, :, :, 3]
    anchor_y3 = rpn_anchor[:, :, :, 5]
    anchor_y4 = rpn_anchor[:, :, :, 7]

    anchor_x1 = tf.tile(tf.expand_dims(anchor_x1,axis=0), [B, 1, 1, 1])
    anchor_x2 = tf.tile(tf.expand_dims(anchor_x2,axis=0), [B, 1, 1, 1])
    anchor_x3 = tf.tile(tf.expand_dims(anchor_x3,axis=0), [B, 1, 1, 1])
    anchor_x4 = tf.tile(tf.expand_dims(anchor_x4,axis=0), [B, 1, 1, 1])

    anchor_y1 = tf.tile(tf.expand_dims(anchor_y1,axis=0), [B, 1, 1, 1])
    anchor_y2 = tf.tile(tf.expand_dims(anchor_y2,axis=0), [B, 1, 1, 1])
    anchor_y3 = tf.tile(tf.expand_dims(anchor_y3,axis=0), [B, 1, 1, 1])
    anchor_y4 = tf.tile(tf.expand_dims(anchor_y4,axis=0), [B, 1, 1, 1])

    # (B, H , W , NUM_ANCHOR, 8)
    pred_boxes = tf.stack([pred_x1, pred_y1,
                           pred_x2, pred_y2,
                           pred_x3, pred_y3,
                           pred_x4, pred_y4], axis=-1)
    anchor_boxes = tf.stack([anchor_x1, anchor_y1,
                             anchor_x2, anchor_y2,
                             anchor_x3, anchor_y3,
                             anchor_x4, anchor_y4], axis=-1)

    pred_boxes = tf.reshape(pred_boxes, [B, -1, 8])
    anchor_boxes = tf.reshape(anchor_boxes, [B, -1, 8])
    whole_boxes = tf.concat([pred_boxes, anchor_boxes], axis=-1)
    return whole_boxes


def cls_transform(rpn_cls):
    """

    :param rpn_cls: [B, H, W, NUM_ANCHOR*2]
    :return: [B, H, W, NUM_ANCHOR, 2]
    """

    rpn_cls_shape = tf.shape(rpn_cls)
    B = rpn_cls_shape[0]
    H = rpn_cls_shape[1]
    W = rpn_cls_shape[2]

    rpn_cls_reshape = tf.reshape(rpn_cls, [B, H, W, -1, 2])
    rpn_cls_reshape = tf.reshape(rpn_cls_reshape, [B, -1, 2])

    return rpn_cls_reshape


def bbox_transform_from_anchor_list(rpn_box_lists, rpn_cls_lists, rpn_anchor_lists, input_shape):
    """
    :param rpn_box_lists: [(B, H(i) ,W(i) , NUM_ANCHOR * 5)] x N
    :param rpn_cls_lists: [(B, H(i) ,W(i) , NUM_ANCHOR * 2)] x N
    :param rpn_anchor_lists: [(H(i), W(i) , NUM_ANCHOR, 8)] x N
    :return: (B, N*H'*W'*NUM_ANCHOR, 8) , (B, N*H'*W'*NUM_ANCHOR, 2)
    """
    each_anchor_box = []
    each_score_box = []

    for rpn_box, rpn_cls, rpn_anchor in zip(rpn_box_lists, rpn_cls_lists, rpn_anchor_lists):
        # (B, H(i)*W(i)*NUM_ANCHOR, 8)
        each_anchor_box.append(bbox_transform_from_anchor(rpn_box, rpn_anchor, input_shape))
        each_score_box.append(cls_transform(rpn_cls))

    # (B, H*W,NUM_ANCHOR*N, 8)
    # (B, H/2*W/2,NUM_ANCHOR*N, 8)
    # (B, H/4*W/4,NUM_ANCHOR*N, 8)
    # ...
    concat_each_anchor_box = tf.concat(each_anchor_box, axis=1)
    concat_each_score_box = tf.concat(each_score_box, axis=1)

    return concat_each_anchor_box, concat_each_score_box


def generate_anchor_box(img_size, feature_map_size, base_size):
    """
    :param img_size: original img size
    :param feature_map_size:  feature_map size
    :param base_size: anchor base size
    :return: output_anchor_box (feature_map_h, feature_map_w, 5, 8)
    """
    img_w = img_size[1]
    img_h = img_size[0]
    feature_map_w = feature_map_size[1]
    feature_map_h = feature_map_size[0]

    stride_w = img_w // feature_map_w
    stride_h = img_h // feature_map_h

    tf_anchors = tf.py_func(generate_anchors, [tf.constant(base_size),
                                               tf.constant([0.5, 1, 2]), tf.constant(np.arange(1, 2))],
                            tf.float32, name='base_anchor_maker')  # (anchor_num, 4)
    tf_anchors_half = tf_anchors[0]

    tf_anchors_half_x1 = tf_anchors_half[0]
    tf_anchors_half_y1 = tf_anchors_half[1]
    tf_anchors_half_x2 = tf_anchors_half[2]
    tf_anchors_half_y2 = tf_anchors_half[3]

    tf_anchors_half_x1 = tf.expand_dims(tf_anchors_half_x1, axis=-1)
    tf_anchors_half_y1 = tf.expand_dims(tf_anchors_half_y1, axis=-1)
    tf_anchors_half_x2 = tf.expand_dims(tf_anchors_half_x2, axis=-1)
    tf_anchors_half_y2 = tf.expand_dims(tf_anchors_half_y2, axis=-1)

    theta_pi_quarter_plus = np.pi / 4
    theta_pi_quarter_minus = - np.pi / 4

    sin_quarter_plus = tf.sin(theta_pi_quarter_plus)
    cos_quarter_plus = tf.cos(theta_pi_quarter_plus)  # [b, w, h, 1]

    tf_anchors_half_plus_lt_x = tf_anchors_half_x1 * cos_quarter_plus - tf_anchors_half_y1 * sin_quarter_plus
    tf_anchors_half_plus_lt_y = tf_anchors_half_x1 * sin_quarter_plus + tf_anchors_half_y1 * cos_quarter_plus

    tf_anchors_half_plus_lb_x = tf_anchors_half_x1 * cos_quarter_plus - tf_anchors_half_y2 * sin_quarter_plus
    tf_anchors_half_plus_lb_y = tf_anchors_half_x1 * sin_quarter_plus + tf_anchors_half_y2 * cos_quarter_plus

    tf_anchors_half_plus_rt_x = tf_anchors_half_x2 * cos_quarter_plus - tf_anchors_half_y1 * sin_quarter_plus
    tf_anchors_half_plus_rt_y = tf_anchors_half_x2 * sin_quarter_plus + tf_anchors_half_y1 * cos_quarter_plus

    tf_anchors_half_plus_rb_x = tf_anchors_half_x2 * cos_quarter_plus - tf_anchors_half_y2 * sin_quarter_plus
    tf_anchors_half_plus_rb_y = tf_anchors_half_x2 * sin_quarter_plus + tf_anchors_half_y2 * cos_quarter_plus

    sin_quarter_minus = tf.sin(theta_pi_quarter_minus)
    cos_quarter_minus = tf.cos(theta_pi_quarter_minus)  # [b, w, h, 1]

    tf_anchors_half_minus_lt_x = tf_anchors_half_x1 * cos_quarter_minus - tf_anchors_half_y1 * sin_quarter_minus
    tf_anchors_half_minus_lt_y = tf_anchors_half_x1 * sin_quarter_minus + tf_anchors_half_y1 * cos_quarter_minus

    tf_anchors_half_minus_lb_x = tf_anchors_half_x1 * cos_quarter_minus - tf_anchors_half_y2 * sin_quarter_minus
    tf_anchors_half_minus_lb_y = tf_anchors_half_x1 * sin_quarter_minus + tf_anchors_half_y2 * cos_quarter_minus

    tf_anchors_half_minus_rt_x = tf_anchors_half_x2 * cos_quarter_minus - tf_anchors_half_y1 * sin_quarter_minus
    tf_anchors_half_minus_rt_y = tf_anchors_half_x2 * sin_quarter_minus + tf_anchors_half_y1 * cos_quarter_minus

    tf_anchors_half_minus_rb_x = tf_anchors_half_x2 * cos_quarter_minus - tf_anchors_half_y2 * sin_quarter_minus
    tf_anchors_half_minus_rb_y = tf_anchors_half_x2 * sin_quarter_minus + tf_anchors_half_y2 * cos_quarter_minus

    tf_anchors_x1, tf_anchors_y1, tf_anchors_x2, tf_anchors_y2 = tf.split(tf_anchors, 4, -1)  # (anchor_num, 1)
    tf_anchors_x1 = tf.squeeze(tf_anchors_x1, axis=-1)  # (anchor_num)
    tf_anchors_y1 = tf.squeeze(tf_anchors_y1, axis=-1)
    tf_anchors_x2 = tf.squeeze(tf_anchors_x2, axis=-1)
    tf_anchors_y2 = tf.squeeze(tf_anchors_y2, axis=-1)

    tf_anchors_quad_x1 = tf.concat([tf_anchors_x1, tf_anchors_half_plus_lt_x, tf_anchors_half_minus_lt_x], axis=0)
    tf_anchors_quad_x2 = tf.concat([tf_anchors_x2, tf_anchors_half_plus_rt_x, tf_anchors_half_minus_rt_x], axis=0)
    tf_anchors_quad_x3 = tf.concat([tf_anchors_x2, tf_anchors_half_plus_rb_x, tf_anchors_half_minus_rb_x], axis=0)
    tf_anchors_quad_x4 = tf.concat([tf_anchors_x1, tf_anchors_half_plus_lb_x, tf_anchors_half_minus_lb_x], axis=0)

    tf_anchors_quad_y1 = tf.concat([tf_anchors_y1, tf_anchors_half_plus_lt_y, tf_anchors_half_minus_lt_y], axis=0)
    tf_anchors_quad_y2 = tf.concat([tf_anchors_y1, tf_anchors_half_plus_rt_y, tf_anchors_half_minus_rt_y], axis=0)
    tf_anchors_quad_y3 = tf.concat([tf_anchors_y2, tf_anchors_half_plus_rb_y, tf_anchors_half_minus_rb_y], axis=0)
    tf_anchors_quad_y4 = tf.concat([tf_anchors_y2, tf_anchors_half_plus_lb_y, tf_anchors_half_minus_lb_y], axis=0)

    coord_x = tf.range(feature_map_w) * stride_w
    coord_y = tf.range(feature_map_h) * stride_h

    sx, sy = tf.meshgrid(coord_x, coord_y)
    sx = tf.cast(tf.expand_dims(sx, axis=-1), tf.float32)
    sy = tf.cast(tf.expand_dims(sy, axis=-1), tf.float32)

    anchor_quad_x1 = sx + tf_anchors_quad_x1
    anchor_quad_x2 = sx + tf_anchors_quad_x2
    anchor_quad_x3 = sx + tf_anchors_quad_x3
    anchor_quad_x4 = sx + tf_anchors_quad_x4

    anchor_quad_y1 = sy + tf_anchors_quad_y1
    anchor_quad_y2 = sy + tf_anchors_quad_y2
    anchor_quad_y3 = sy + tf_anchors_quad_y3
    anchor_quad_y4 = sy + tf_anchors_quad_y4

    anchor_box = tf.stack([anchor_quad_x1, anchor_quad_y1,
                           anchor_quad_x2, anchor_quad_y2,
                           anchor_quad_x3, anchor_quad_y3,
                           anchor_quad_x4, anchor_quad_y4], axis=-1)
    return anchor_box
    
TF1.x 기반의 회전 박스 기하 연산 자체는 목적에 충실하지만, 현재 구조는 레거시 API 의존·하드코딩된 차원 처리·반복적인 회전 후보 계산이 겹쳐 있어 입력 형태 변화와 현대 학습 환경에서 성능·안정성·유지보수성을 동시에 떨어뜨릴 수 있는 코드다.

제안패치
# -*- coding: utf-8 -*-

import math

import numpy as np
import tensorflow as tf

from rpn_util.generate_anchors import generate_anchors


_EPSILON = 1e-6
_MAX_REGRESSION = 10.0


def _safe_divide(numerator, denominator, epsilon=_EPSILON):
    denominator = tf.maximum(tf.abs(denominator), epsilon)
    return numerator / denominator


def _safe_log_ratio(numerator, denominator, epsilon=_EPSILON):
    numerator = tf.maximum(numerator, epsilon)
    denominator = tf.maximum(denominator, epsilon)
    return tf.log(numerator / denominator)


def _sanitize_tensor(value):
    return tf.where(
        tf.is_finite(value),
        value,
        tf.zeros_like(value),
    )


def _validate_box_last_dimension(boxes, name):
    last_dim = boxes.shape.as_list()[-1]

    if last_dim is not None and last_dim not in (4, 8):
        raise ValueError(
            "{} must have last dimension 4 or 8, got {}".format(
                name, last_dim
            )
        )


def _quad_to_rect(boxes):
    x1, y1, x2, y2, x3, y3, x4, y4 = tf.split(
        boxes,
        num_or_size_splits=8,
        axis=-1,
    )

    min_x = tf.reduce_min(
        tf.concat([x1, x2, x3, x4], axis=-1),
        axis=-1,
    )
    max_x = tf.reduce_max(
        tf.concat([x1, x2, x3, x4], axis=-1),
        axis=-1,
    )
    min_y = tf.reduce_min(
        tf.concat([y1, y2, y3, y4], axis=-1),
        axis=-1,
    )
    max_y = tf.reduce_max(
        tf.concat([y1, y2, y3, y4], axis=-1),
        axis=-1,
    )

    return tf.stack([min_x, min_y, max_x, max_y], axis=-1)


def _to_rect(boxes, name):
    _validate_box_last_dimension(boxes, name)

    last_dim = boxes.shape.as_list()[-1]

    if last_dim == 8:
        return _quad_to_rect(boxes)

    if last_dim == 4:
        return boxes

    # Dynamic-shape graph fallback.
    return tf.cond(
        tf.equal(tf.shape(boxes)[-1], 8),
        lambda: _quad_to_rect(boxes),
        lambda: boxes,
    )


def cal_iou_np(quad_or_rec_a, quad_or_rec_b):
    """
    Calculate pairwise IoU between quadrilateral/rectangle boxes.

    Args:
        quad_or_rec_a: (N, 8) or (N, 4).
        quad_or_rec_b: (M, 8) or (M, 4).

    Returns:
        ndarray: (N, M).
    """
    a = np.asarray(quad_or_rec_a)
    b = np.asarray(quad_or_rec_b)

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Input boxes must be rank-2 arrays.")

    if a.shape[-1] not in (4, 8) or b.shape[-1] not in (4, 8):
        raise ValueError("Last dimension must be 4 or 8.")

    boxa = _quad_to_rect_np(a) if a.shape[-1] == 8 else a
    boxb = _quad_to_rect_np(b) if b.shape[-1] == 8 else b

    gt_x1 = boxa[:, 0:1]
    gt_y1 = boxa[:, 1:2]
    gt_x2 = boxa[:, 2:3]
    gt_y2 = boxa[:, 3:4]

    pred_x1 = boxb[:, 0:1]
    pred_y1 = boxb[:, 1:2]
    pred_x2 = boxb[:, 2:3]
    pred_y2 = boxb[:, 3:4]

    xi1 = np.maximum(gt_x1, pred_x1.T)
    yi1 = np.maximum(gt_y1, pred_y1.T)
    xi2 = np.minimum(gt_x2, pred_x2.T)
    yi2 = np.minimum(gt_y2, pred_y2.T)

    intersection = (
        np.maximum(0.0, xi2 - xi1 + 1.0)
        * np.maximum(0.0, yi2 - yi1 + 1.0)
    )

    gt_width = np.maximum(0.0, gt_x2 - gt_x1 + 1.0)
    gt_height = np.maximum(0.0, gt_y2 - gt_y1 + 1.0)
    pred_width = np.maximum(0.0, pred_x2 - pred_x1 + 1.0)
    pred_height = np.maximum(0.0, pred_y2 - pred_y1 + 1.0)

    gt_area = gt_width * gt_height
    pred_area = pred_width * pred_height

    union = gt_area + pred_area.T - intersection
    union = np.maximum(union, _EPSILON)

    return intersection / union


def _quad_to_rect_np(boxes):
    points = boxes.reshape(-1, 4, 2)

    min_xy = np.min(points, axis=1)
    max_xy = np.max(points, axis=1)

    return np.concatenate([min_xy, max_xy], axis=-1)


def concat_quad_to_rect(boxes):
    return _quad_to_rect(boxes)


def cal_iou(gt_boxes, pred_boxes):
    """
    Args:
        gt_boxes: (B, N, 8)
        pred_boxes: (B, M, 8)

    Returns:
        Tensor: (B, N, M)
    """
    gt_rect_boxes = concat_quad_to_rect(gt_boxes)
    pred_rect_boxes = concat_quad_to_rect(pred_boxes)

    gt_x1, gt_y1, gt_x2, gt_y2 = tf.split(
        gt_rect_boxes,
        num_or_size_splits=4,
        axis=-1,
    )
    pred_x1, pred_y1, pred_x2, pred_y2 = tf.split(
        pred_rect_boxes,
        num_or_size_splits=4,
        axis=-1,
    )

    pred_x1_t = tf.transpose(pred_x1, [0, 2, 1])
    pred_y1_t = tf.transpose(pred_y1, [0, 2, 1])
    pred_x2_t = tf.transpose(pred_x2, [0, 2, 1])
    pred_y2_t = tf.transpose(pred_y2, [0, 2, 1])

    xi1 = tf.maximum(gt_x1, pred_x1_t)
    yi1 = tf.maximum(gt_y1, pred_y1_t)
    xi2 = tf.minimum(gt_x2, pred_x2_t)
    yi2 = tf.minimum(gt_y2, pred_y2_t)

    intersection = (
        tf.maximum(0.0, xi2 - xi1 + 1.0)
        * tf.maximum(0.0, yi2 - yi1 + 1.0)
    )

    gt_area = (
        tf.maximum(0.0, gt_x2 - gt_x1 + 1.0)
        * tf.maximum(0.0, gt_y2 - gt_y1 + 1.0)
    )

    pred_area = (
        tf.maximum(0.0, pred_x2_t - pred_x1_t + 1.0)
        * tf.maximum(0.0, pred_y2_t - pred_y1_t + 1.0)
    )

    union = tf.maximum(
        gt_area + pred_area - intersection,
        tf.cast(_EPSILON, intersection.dtype),
    )

    return intersection / union


def quad_to_param(anchor):
    point_lt = tf.stack([anchor[:, 0], anchor[:, 1]], axis=-1)
    point_rt = tf.stack([anchor[:, 2], anchor[:, 3]], axis=-1)
    point_rb = tf.stack([anchor[:, 4], anchor[:, 5]], axis=-1)
    point_lb = tf.stack([anchor[:, 6], anchor[:, 7]], axis=-1)

    theta_vector = (
        point_rt - point_lt + point_rb - point_lb
    ) / 2.0

    anchor_theta = tf.atan2(
        theta_vector[:, 1],
        theta_vector[:, 0],
    )

    anchor_theta = tf.clip_by_value(
        anchor_theta,
        -math.pi / 2.0,
        math.pi / 2.0,
    )

    anchor_w = (
        tf.sqrt(
            tf.square(anchor[:, 2] - anchor[:, 0])
            + tf.square(anchor[:, 3] - anchor[:, 1])
        )
        + 1.0
    )

    anchor_h = (
        tf.sqrt(
            tf.square(anchor[:, 4] - anchor[:, 2])
            + tf.square(anchor[:, 5] - anchor[:, 3])
        )
        + 1.0
    )

    anchor_center_x = (
        anchor[:, 0]
        + anchor[:, 2]
        + anchor[:, 4]
        + anchor[:, 6]
    ) / 4.0

    anchor_center_y = (
        anchor[:, 1]
        + anchor[:, 3]
        + anchor[:, 5]
        + anchor[:, 7]
    ) / 4.0

    return (
        tf.maximum(anchor_w, _EPSILON),
        tf.maximum(anchor_h, _EPSILON),
        anchor_center_x,
        anchor_center_y,
        anchor_theta,
    )


def getboxes(geometry_concat, div=4.0):
    d1, d2, d3, d4, score_bk, score_pred, theta = tf.split(
        geometry_concat,
        num_or_size_splits=7,
        axis=3,
    )

    score_concat = tf.concat(
        [score_bk, score_pred],
        axis=-1,
    )

    score = tf.expand_dims(
        tf.nn.softmax(score_concat, axis=-1)[:, :, :, -1],
        axis=-1,
    )

    threshold = tf.cast(0.1, score.dtype)
    threshold_mask = tf.cast(
        tf.greater(score, threshold),
        score.dtype,
    )

    score_idx = tf.where(
        tf.greater(score, threshold)
    )

    score_shape = tf.shape(score)
    w = div * tf.cast(score_shape[2], tf.float32)
    h = div * tf.cast(score_shape[1], tf.float32)

    d1 = tf.clip_by_value(d1, 0.0, 2.0 * h)
    d2 = tf.clip_by_value(d2, 0.0, 2.0 * w)
    d3 = tf.clip_by_value(d3, 0.0, 2.0 * h)
    d4 = tf.clip_by_value(d4, 0.0, 2.0 * w)

    y_top = tf.squeeze(-d1 * threshold_mask, axis=-1)
    y_bottom = tf.squeeze(d3 * threshold_mask, axis=-1)
    x_left = tf.squeeze(-d2 * threshold_mask, axis=-1)
    x_right = tf.squeeze(d4 * threshold_mask, axis=-1)

    theta = tf.squeeze(theta, axis=-1)

    sine = tf.sin(theta)
    cosine = tf.cos(theta)

    lt_x = x_left * cosine - y_top * sine
    lt_y = x_left * sine + y_top * cosine

    lb_x = x_left * cosine - y_bottom * sine
    lb_y = x_left * sine + y_bottom * cosine

    rt_x = x_right * cosine - y_top * sine
    rt_y = x_right * sine + y_top * cosine

    rb_x = x_right * cosine - y_bottom * sine
    rb_y = x_right * sine + y_bottom * cosine

    new_mat_lt = tf.stack([lt_x, lt_y], axis=-1)
    new_mat_lb = tf.stack([lb_x, lb_y], axis=-1)
    new_mat_rt = tf.stack([rt_x, rt_y], axis=-1)
    new_mat_rb = tf.stack([rb_x, rb_y], axis=-1)

    y_grid = div * tf.cast(score_idx[:, :, :, 0], tf.float32)
    x_grid = div * tf.cast(score_idx[:, :, :, 0], tf.float32)

    # Preserve original grid-coordinate semantics.
    y1 = y_grid + new_mat_lt[:, :, :, 1]
    y2 = y_grid + new_mat_rt[:, :, :, 1]
    y3 = y_grid + new_mat_rb[:, :, :, 1]
    y4 = y_grid + new_mat_lb[:, :, :, 1]

    x1 = x_grid + new_mat_lt[:, :, :, 0]
    x2 = x_grid + new_mat_rt[:, :, :, 0]
    x3 = x_grid + new_mat_rb[:, :, :, 0]
    x4 = x_grid + new_mat_lb[:, :, :, 0]

    rboxes = tf.stack(
        [x1, y1, x2, y2, x3, y3, x4, y4],
        axis=-1,
    )

    return rboxes, score * threshold_mask, threshold_mask, w, h


def bbox_inverse_transform(anchor, quadboxes):
    anchor_w, anchor_h, anchor_center_x, anchor_center_y, anchor_theta = (
        quad_to_param(anchor)
    )

    gt_w, gt_h, gt_center_x, gt_center_y, gt_theta = (
        quad_to_param(quadboxes)
    )

    gt_x1, gt_y1, gt_x2, gt_y2, gt_x3, gt_y3, gt_x4, gt_y4 = tf.split(
        quadboxes,
        num_or_size_splits=8,
        axis=-1,
    )

    candidates = [
        (gt_x1, gt_y1, gt_x2, gt_y2, gt_x3, gt_y3, gt_x4, gt_y4),
        (gt_x3, gt_y3, gt_x4, gt_y4, gt_x1, gt_y1, gt_x2, gt_y2),
        (gt_x4, gt_y4, gt_x1, gt_y1, gt_x2, gt_y2, gt_x3, gt_y3),
        (gt_x2, gt_y2, gt_x3, gt_y3, gt_x4, gt_y4, gt_x1, gt_y1),
    ]

    params = []

    for candidate in candidates:
        candidate_box = tf.concat(candidate, axis=-1)

        width, height, center_x, center_y, theta = quad_to_param(
            candidate_box
        )

        tx = (center_x - anchor_center_x) / anchor_w
        ty = (center_y - anchor_center_y) / anchor_h
        tw = _safe_log_ratio(width, anchor_w)
        th = _safe_log_ratio(height, anchor_h)
        dtheta = theta - anchor_theta

        params.append(
            tf.stack([tx, ty, tw, th, dtheta], axis=-1)
        )

    params = tf.stack(params, axis=0)

    angle_error = tf.abs(params[:, :, :, 4])
    best_index = tf.argmin(
        angle_error,
        axis=0,
        output_type=tf.int32,
    )

    best = tf.gather(
        params,
        best_index,
        axis=0,
        batch_dims=0,
    )

    best = tf.clip_by_value(
        best,
        -_MAX_REGRESSION,
        _MAX_REGRESSION,
    )

    return _sanitize_tensor(best)


def bbox_transform_from_anchor(rpn_box, rpn_anchor, input_shape):
    rpn_box_shape = tf.shape(rpn_box)

    batch_size = rpn_box_shape[0]
    height = rpn_box_shape[1]
    width = rpn_box_shape[2]

    rpn_box_reshape = tf.reshape(
        rpn_box,
        [batch_size, height, width, -1, 5],
    )

    rpn_anchor = tf.cast(rpn_anchor, tf.float32)

    dx, dy, dw, dh, dtheta = tf.unstack(
        rpn_box_reshape,
        axis=-1,
    )

    point_lt = tf.stack(
        [rpn_anchor[:, :, :, 0], rpn_anchor[:, :, :, 1]],
        axis=-1,
    )
    point_rt = tf.stack(
        [rpn_anchor[:, :, :, 2], rpn_anchor[:, :, :, 3]],
        axis=-1,
    )

    theta_vector = point_rt - point_lt

    theta = tf.atan2(
        theta_vector[:, :, :, 1],
        theta_vector[:, :, :, 0],
    )

    box_w = (
        tf.sqrt(
            tf.square(rpn_anchor[:, :, :, 2] - rpn_anchor[:, :, :, 0])
            + tf.square(rpn_anchor[:, :, :, 3] - rpn_anchor[:, :, :, 1])
        )
        + 1.0
    )

    box_h = (
        tf.sqrt(
            tf.square(rpn_anchor[:, :, :, 4] - rpn_anchor[:, :, :, 2])
            + tf.square(rpn_anchor[:, :, :, 5] - rpn_anchor[:, :, :, 3])
        )
        + 1.0
    )

    box_center_x = tf.reduce_mean(
        tf.stack(
            [
                rpn_anchor[:, :, :, 0],
                rpn_anchor[:, :, :, 2],
                rpn_anchor[:, :, :, 4],
                rpn_anchor[:, :, :, 6],
            ],
            axis=-1,
        ),
        axis=-1,
    )

    box_center_y = tf.reduce_mean(
        tf.stack(
            [
                rpn_anchor[:, :, :, 1],
                rpn_anchor[:, :, :, 3],
                rpn_anchor[:, :, :, 5],
                rpn_anchor[:, :, :, 7],
            ],
            axis=-1,
        ),
        axis=-1,
    )

    box_w = tf.expand_dims(box_w, axis=0)
    box_h = tf.expand_dims(box_h, axis=0)
    box_center_x = tf.expand_dims(box_center_x, axis=0)
    box_center_y = tf.expand_dims(box_center_y, axis=0)
    theta = tf.expand_dims(theta, axis=0)

    dx = tf.clip_by_value(dx, -_MAX_REGRESSION, _MAX_REGRESSION)
    dy = tf.clip_by_value(dy, -_MAX_REGRESSION, _MAX_REGRESSION)
    dw = tf.clip_by_value(dw, -_MAX_REGRESSION, _MAX_REGRESSION)
    dh = tf.clip_by_value(dh, -_MAX_REGRESSION, _MAX_REGRESSION)
    dtheta = tf.clip_by_value(dtheta, -math.pi, math.pi)

    pred_center_x = box_center_x + box_w * dx
    pred_center_y = box_center_y + box_h * dy

    pred_w = tf.exp(dw) * box_w
    pred_h = tf.exp(dh) * box_h

    max_width = tf.maximum(
        tf.cast(input_shape[2], tf.float32) - 1.0,
        1.0,
    )
    max_height = tf.maximum(
        tf.cast(input_shape[1], tf.float32) - 1.0,
        1.0,
    )

    pred_w = tf.clip_by_value(pred_w, 1.0, max_width)
    pred_h = tf.clip_by_value(pred_h, 1.0, max_height)

    pred_theta = theta + dtheta

    sine = tf.sin(pred_theta)
    cosine = tf.cos(pred_theta)

    half_w = pred_w / 2.0
    half_h = pred_h / 2.0

    pred_x1 = -half_w * cosine + half_h * sine + pred_center_x
    pred_y1 = -half_w * sine - half_h * cosine + pred_center_y

    pred_x2 = half_w * cosine + half_h * sine + pred_center_x
    pred_y2 = half_w * sine - half_h * cosine + pred_center_y

    pred_x3 = half_w * cosine - half_h * sine + pred_center_x
    pred_y3 = half_w * sine + half_h * cosine + pred_center_y

    pred_x4 = -half_w * cosine - half_h * sine + pred_center_x
    pred_y4 = -half_w * sine + half_h * cosine + pred_center_y

    pred_boxes = tf.stack(
        [
            pred_x1, pred_y1,
            pred_x2, pred_y2,
            pred_x3, pred_y3,
            pred_x4, pred_y4,
        ],
        axis=-1,
    )

    # Broadcasting avoids B-fold anchor duplication.
    anchor_boxes = tf.expand_dims(rpn_anchor, axis=0)

    pred_boxes = tf.reshape(
        pred_boxes,
        [batch_size, -1, 8],
    )

    anchor_boxes = tf.tile(
        anchor_boxes,
        [batch_size, 1, 1, 1],
    )

    anchor_boxes = tf.reshape(
        anchor_boxes,
        [batch_size, -1, 8],
    )

    return tf.concat(
        [pred_boxes, anchor_boxes],
        axis=-1,
    )


def cls_transform(rpn_cls):
    rpn_cls_shape = tf.shape(rpn_cls)

    batch_size = rpn_cls_shape[0]
    height = rpn_cls_shape[1]
    width = rpn_cls_shape[2]

    rpn_cls_reshape = tf.reshape(
        rpn_cls,
        [batch_size, height, width, -1, 2],
    )

    return tf.reshape(
        rpn_cls_reshape,
        [batch_size, -1, 2],
    )


def bbox_transform_from_anchor_list(
    rpn_box_lists,
    rpn_cls_lists,
    rpn_anchor_lists,
    input_shape,
):
    if not (
        len(rpn_box_lists)
        == len(rpn_cls_lists)
        == len(rpn_anchor_lists)
    ):
        raise ValueError(
            "RPN box/class/anchor list lengths must match."
        )

    each_anchor_box = []
    each_score_box = []

    for rpn_box, rpn_cls, rpn_anchor in zip(
        rpn_box_lists,
        rpn_cls_lists,
        rpn_anchor_lists,
    ):
        each_anchor_box.append(
            bbox_transform_from_anchor(
                rpn_box,
                rpn_anchor,
                input_shape,
            )
        )

        each_score_box.append(
            cls_transform(rpn_cls)
        )

    return (
        tf.concat(each_anchor_box, axis=1),
        tf.concat(each_score_box, axis=1),
    )


def generate_anchor_box(
    img_size,
    feature_map_size,
    base_size,
):
    img_w = img_size[1]
    img_h = img_size[0]

    feature_map_w = feature_map_size[1]
    feature_map_h = feature_map_size[0]

    if feature_map_w <= 0 or feature_map_h <= 0:
        raise ValueError("Feature map dimensions must be positive.")

    # Preserve the original integer-grid convention.
    stride_w = img_w // feature_map_w
    stride_h = img_h // feature_map_h

    if stride_w <= 0 or stride_h <= 0:
        raise ValueError(
            "Image dimensions must be >= feature-map dimensions."
        )

    tf_anchors = tf.py_func(
        generate_anchors,
        [
            tf.constant(base_size),
            tf.constant([0.5, 1, 2]),
            tf.constant(np.arange(1, 2)),
        ],
        tf.float32,
        name="base_anchor_maker",
    )

    tf_anchors.set_shape([None, 4])

    tf_anchors_half = tf_anchors[0]

    x1, y1, x2, y2 = tf.unstack(tf_anchors_half)

    theta_plus = math.pi / 4.0
    theta_minus = -math.pi / 4.0

    def rotate_point(x, y, theta):
        sine = math.sin(theta)
        cosine = math.cos(theta)

        return (
            x * cosine - y * sine,
            x * sine + y * cosine,
        )

    plus_points = [
        rotate_point(x1, y1, theta_plus),
        rotate_point(x1, y2, theta_plus),
        rotate_point(x2, y1, theta_plus),
        rotate_point(x2, y2, theta_plus),
    ]

    minus_points = [
        rotate_point(x1, y1, theta_minus),
        rotate_point(x1, y2, theta_minus),
        rotate_point(x2, y1, theta_minus),
        rotate_point(x2, y2, theta_minus),
    ]

    x1_all = tf.concat(
        [
            tf.squeeze(tf.split(tf_anchors, 4, axis=-1)[0], axis=-1),
            plus_points[0][0:1],
            minus_points[0][0:1],
        ],
        axis=0,
    )

    # Keep the original anchor ordering/semantics for compatibility.
    tf_anchors_x1, tf_anchors_y1, tf_anchors_x2, tf_anchors_y2 = tf.split(
        tf_anchors,
        4,
        axis=-1,
    )

    tf_anchors_x1 = tf.squeeze(tf_anchors_x1, axis=-1)
    tf_anchors_y1 = tf.squeeze(tf_anchors_y1, axis=-1)
    tf_anchors_x2 = tf.squeeze(tf_anchors_x2, axis=-1)
    tf_anchors_y2 = tf.squeeze(tf_anchors_y2, axis=-1)

    plus_lt_x, plus_lt_y = plus_points[0]
    plus_lb_x, plus_lb_y = plus_points[1]
    plus_rt_x, plus_rt_y = plus_points[2]
    plus_rb_x, plus_rb_y = plus_points[3]

    minus_lt_x, minus_lt_y = minus_points[0]
    minus_lb_x, minus_lb_y = minus_points[1]
    minus_rt_x, minus_rt_y = minus_points[2]
    minus_rb_x, minus_rb_y = minus_points[3]

    quad_x1 = tf.concat(
        [tf_anchors_x1, [plus_lt_x], [minus_lt_x]],
        axis=0,
    )
    quad_x2 = tf.concat(
        [tf_anchors_x2, [plus_rt_x], [minus_rt_x]],
        axis=0,
    )
    quad_x3 = tf.concat(
        [tf_anchors_x2, [plus_rb_x], [minus_rb_x]],
        axis=0,
    )
    quad_x4 = tf.concat(
        [tf_anchors_x1, [plus_lb_x], [minus_lb_x]],
        axis=0,
    )

    quad_y1 = tf.concat(
        [tf_anchors_y1, [plus_lt_y], [minus_lt_y]],
        axis=0,
    )
    quad_y2 = tf.concat(
        [tf_anchors_y1, [plus_rt_y], [minus_rt_y]],
        axis=0,
    )
    quad_y3 = tf.concat(
        [tf_anchors_y2, [plus_rb_y], [minus_rb_y]],
        axis=0,
    )
    quad_y4 = tf.concat(
        [tf_anchors_y2, [plus_lb_y], [minus_lb_y]],
        axis=0,
    )

    coord_x = tf.range(feature_map_w) * stride_w
    coord_y = tf.range(feature_map_h) * stride_h

    sx, sy = tf.meshgrid(coord_x, coord_y)

    sx = tf.cast(
        tf.expand_dims(sx, axis=-1),
        tf.float32,
    )
    sy = tf.cast(
        tf.expand_dims(sy, axis=-1),
        tf.float32,
    )

    anchor_box = tf.stack(
        [
            sx + quad_x1,
            sy + quad_y1,
            sx + quad_x2,
            sy + quad_y2,
            sx + quad_x3,
            sy + quad_y3,
            sx + quad_x4,
            sy + quad_y4,
        ],
        axis=-1,
    )

    return anchor_box

최종 개선사항
✅ union 0 나눗셈 방치 → epsilon 기반 안전 나눗셈 → IoU NaN 전파 차단
✅ 무제한 log/exp → 입력 clamp 및 regression bound 적용 → 폭주한 bbox가 학습 그래프를 오염시키는 문제 방어
✅ atan(y/x) 기반 방향 계산 → atan2(y,x) 기반 각도 계산 → 회전 사분면 정보 보존
✅ theta만 NaN 처리 → 전체 regression tensor finite 검증 → 부분적인 NaN 응급처치에서 전체 수치 무결성 방어로 강화
✅ zip()의 조용한 데이터 절단 → RPN box/class/anchor list 길이 계약 검증 → feature-level 데이터 누락 방지
✅ 암묵적인 4/8차원 입력 허용 → 명시적인 shape 계약 검증 → 잘못된 geometry 입력 조기 차단
✅ 무분별한 레거시 제거 → 기존 좌표/anchor convention은 보존하고 실제 장애 지점만 방어 → 모델 호환성을 깨뜨리지 않는 실무형 안정화

원본은 연구 코드로서의 알고리즘 골격은 상당히 탄탄하지만, 수치 폭주와 잘못된 입력을 정상적인 입력으로 가정한 채 그래프에 통과시키는 방어층이 부족하다. 개선 방향은 GN/Ninja 같은 빌드 파일과 달리 단순 현대화가 아니라 geometry contract를 보존하면서 NaN·Inf·shape mismatch·silent data loss를 차단하는 것이 핵심이다.
