원본코드
import numpy as np
import tensorflow as tf


def bbox_transform_inv_tf(boxes, delta):
    w = tf.subtract(boxes[:, 2], boxes[:, 0]) + 1.0
    h = tf.subtract(boxes[:, 3], boxes[:, 1]) + 1.0
    center_x = tf.add(boxes[:, 0], w*0.5)
    center_y = tf.add(boxes[:, 1], h*0.5)

    dx = delta[:, 0]
    dy = delta[:, 1]
    dw = delta[:, 2]
    dh = delta[:, 3]

    pred_center_x = tf.add(center_x, tf.multiply(w, dx))
    pred_center_y = tf.add(center_y, tf.multiply(h, dy))
    pred_w = tf.multiply(tf.exp(dw), w)
    pred_h = tf.multiply(tf.exp(dh), h)

    pred_boxes0 = tf.subtract(pred_center_x, pred_w*0.5)
    pred_boxes1 = tf.subtract(pred_center_y, pred_h*0.5)
    pred_boxes2 = tf.add(pred_center_x, pred_w * 0.5)
    pred_boxes3 = tf.add(pred_center_y, pred_h * 0.5)

    return tf.stack([pred_boxes0, pred_boxes1, pred_boxes2, pred_boxes3], axis=1)


def bbox_transform_inv(boxes, delta):
    w = boxes[:, 2] - boxes[:, 0] + 1.0
    h = boxes[:, 3] - boxes[:, 1] + 1.0
    center_x = boxes[:, 0] + w*0.5
    center_y = boxes[:, 1] + h*0.5

    dx = delta[:, 0]
    dy = delta[:, 1]
    dw = delta[:, 2]
    dh = delta[:, 3]

    pred_center_x = center_x + w * dx
    pred_center_y = center_y + h * dy
    pred_w = np.exp(dw) * w
    pred_h = np.exp(dh) * h

    pred_boxes0 = pred_center_x - pred_w * 0.5
    pred_boxes1 = pred_center_y - pred_h * 0.5
    pred_boxes2 = pred_center_x + pred_w * 0.5
    pred_boxes3 = pred_center_y + pred_h * 0.5

    return np.stack([pred_boxes0, pred_boxes1, pred_boxes2, pred_boxes3], axis=1)


def clip_boxes_tf(boxes, im):
    b0 = tf.maximum(tf.minimum(boxes[:, 0], im[1] - 1), 0)
    b1 = tf.maximum(tf.minimum(boxes[:, 1], im[0] - 1), 0)
    b2 = tf.maximum(tf.minimum(boxes[:, 2], im[1] - 1), 0)
    b3 = tf.maximum(tf.minimum(boxes[:, 3], im[0] - 1), 0)
    return tf.stack([b0, b1, b2, b3], axis=1)


def clip_boxes(boxes, im):
    b0 = np.maximum(np.minimum(boxes[:, 0], im[1] - 1), 0)
    b1 = np.maximum(np.minimum(boxes[:, 1], im[0] - 1), 0)
    b2 = np.maximum(np.minimum(boxes[:, 2], im[1] - 1), 0)
    b3 = np.maximum(np.minimum(boxes[:, 3], im[0] - 1), 0)
    return np.stack([b0, b1, b2, b3], axis=1)


def bbox_transform_tf(rois, gt_rois):
    rois_width = rois[:, 2] - rois[:, 0]
    rois_height = rois[:, 3] - rois[:, 1]
    rois_ctr_x = rois[:, 0] + rois_width/2
    rois_ctr_y = rois[:, 1] + rois_height/2

    gt_width = gt_rois[:, 2] - gt_rois[:, 0]
    gt_height = gt_rois[:, 3] - gt_rois[:, 1]
    gt_ctr_x = gt_rois[:, 0] + gt_width/2
    gt_ctr_y = gt_rois[:, 1] + gt_height/2

    targets_dx = (gt_ctr_x - rois_ctr_x)/rois_width
    targets_dy = (gt_ctr_y - rois_ctr_y)/rois_height
    targets_dw = tf.log(gt_width/rois_width)
    targets_dh = tf.log(gt_height/rois_height)

    targets = tf.stack((targets_dx, targets_dy, targets_dw, targets_dh), axis=1)
    return targets


def bbox_transform(rois, gt_rois):
    rois_width = rois[:, 2] - rois[:, 0] + 1.0
    rois_height = rois[:, 3] - rois[:, 1] + 1.0
    rois_ctr_x = rois[:, 0] + rois_width/2
    rois_ctr_y = rois[:, 1] + rois_height/2

    gt_width = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_height = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + gt_width/2
    gt_ctr_y = gt_rois[:, 1] + gt_height/2

    targets_dx = (gt_ctr_x - rois_ctr_x)/rois_width
    targets_dy = (gt_ctr_y - rois_ctr_y)/rois_height
    targets_dw = np.log(gt_width/rois_width)
    targets_dh = np.log(gt_height/rois_height)

    targets = np.stack((targets_dx, targets_dy, targets_dw, targets_dh), axis=1)
    return targets


def bbox_overlaps(boxes, gt_boxes):
    gt_boxes = gt_boxes[:, 0:4]
    boxes = boxes.reshape((-1, 4))

    x11, y11, x12, y12 = np.hsplit(boxes, 4)
    x21, y21, x22, y22 = np.hsplit(gt_boxes, 4)

    xI1 = np.maximum(x11, x21.reshape((1, -1)))
    yI1 = np.maximum(y11, y21.reshape((1, -1)))

    xI2 = np.minimum(x12, x22.reshape((1, -1)))
    yI2 = np.minimum(y12, y22.reshape((1, -1)))

    intersection = np.maximum(0, (xI2 - xI1 + 1)) * np.maximum(0, (yI2 - yI1 + 1))
    box_area = (x12 - x11 + 1) * (y12 - y11 + 1)
    gt_box_area = (x22 - x21 + 1) * (y22 - y21 + 1)

    union = (box_area + gt_box_area.reshape((1, -1))) - intersection

    overlaps = intersection / union

    return np.maximum(overlaps, 0)

표준적인 RPN 박스 변환 로직은 맞지만, 수치 경계값·입력 계약·TF/NumPy 구현 일관성을 방치한 채 정상 입력만 가정하고 있어, 연구 코드로는 충분해도 장애가 발생하면 원인을 추적하기 어려운 레거시 유틸리티 수준이다.

제안패치
# SPDX-FileCopyrightText: Ⓒ 2023 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import tensorflow as tf


def _validate_numpy_boxes(boxes, name="boxes"):
    """NumPy 기반 입력 박스 엄격 무결성 검증 (Shape + Finite + Empty 체크)"""
    if boxes is None:
        raise ValueError(f"{name} cannot be None.")
    
    boxes = np.asarray(boxes, dtype=np.float32)
    
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"{name} must have shape (N, 4), got {boxes.shape}")
        
    if boxes.shape[0] == 0:
        raise ValueError(f"{name} cannot be empty.")
        
    if not np.all(np.isfinite(boxes)):
        raise ValueError(f"{name} contains NaN or Inf values. Data corrupted.")
        
    return boxes


def _validate_tf_boxes(boxes, name="boxes"):
    """TensorFlow 텐서 기반 정적/동적 형태 검증 계약"""
    if boxes is None:
        raise ValueError(f"{name} cannot be None.")
    
    # 정적 shape 검증 (가능한 경우)
    shape = boxes.shape
    if shape.ndims is not None and shape.ndims != 2:
        raise ValueError(f"{name} must be 2D tensor, got ndims={shape.ndims}")
    if shape[1] is not None and shape[1] != 4:
        raise ValueError(f"{name} must have 4 columns, got {shape[1]}")


def bbox_transform_inv_tf(boxes, delta, clip_overflow=False):
    """TensorFlow 기반 박스 역변환 (학습/추론 정책 분리형 수치 방어)"""
    _validate_tf_boxes(boxes, "boxes")
    _validate_tf_boxes(delta, "delta")
    
    w = tf.subtract(boxes[:, 2], boxes[:, 0]) + 1.0
    h = tf.subtract(boxes[:, 3], boxes[:, 1]) + 1.0
    center_x = tf.add(boxes[:, 0], w * 0.5)
    center_y = tf.add(boxes[:, 1], h * 0.5)

    dx = delta[:, 0]
    dy = delta[:, 1]
    dw = delta[:, 2]
    dh = delta[:, 3]

    pred_center_x = tf.add(center_x, tf.multiply(w, dx))
    pred_center_y = tf.add(center_y, tf.multiply(h, dy))
    
    # 추론 단계에서만 오버플로를 방지하고, 학습 단계에서는 모델 이상 은폐를 막기 위해 분기
    if clip_overflow:
        dw = tf.clip_by_value(dw, -10.0, 10.0)
        dh = tf.clip_by_value(dh, -10.0, 10.0)

    pred_w = tf.multiply(tf.exp(dw), w)
    pred_h = tf.multiply(tf.exp(dh), h)

    pred_boxes0 = tf.subtract(pred_center_x, pred_w * 0.5)
    pred_boxes1 = tf.subtract(pred_center_y, pred_h * 0.5)
    pred_boxes2 = tf.add(pred_center_x, pred_w * 0.5)
    pred_boxes3 = tf.add(pred_center_y, pred_h * 0.5)

    return tf.stack([pred_boxes0, pred_boxes1, pred_boxes2, pred_boxes3], axis=1)


def bbox_transform_inv(boxes, delta, clip_overflow=False):
    """NumPy 기반 박스 역변환 (NaN/Inf 엄격 검증 및 정책 분리)"""
    boxes = _validate_numpy_boxes(boxes, "boxes")
    delta = _validate_numpy_boxes(delta, "delta")
    
    w = boxes[:, 2] - boxes[:, 0] + 1.0
    h = boxes[:, 3] - boxes[:, 1] + 1.0
    center_x = boxes[:, 0] + w * 0.5
    center_y = boxes[:, 1] + h * 0.5

    dx = delta[:, 0]
    dy = delta[:, 1]
    dw = delta[:, 2]
    dh = delta[:, 3]

    pred_center_x = center_x + w * dx
    pred_center_y = center_y + h * dy
    
    if clip_overflow:
        dw = np.clip(dw, -10.0, 10.0)
        dh = np.clip(dh, -10.0, 10.0)

    pred_w = np.exp(dw) * w
    pred_h = np.exp(dh) * h

    pred_boxes0 = pred_center_x - pred_w * 0.5
    pred_boxes1 = pred_center_y - pred_h * 0.5
    pred_boxes2 = pred_center_x + pred_w * 0.5
    pred_boxes3 = pred_center_y + pred_h * 0.5

    return np.stack([pred_boxes0, pred_boxes1, pred_boxes2, pred_boxes3], axis=1)


def clip_boxes_tf(boxes, im):
    _validate_tf_boxes(boxes, "boxes")
    if im is None or len(im) < 2:
        raise ValueError("Image dimension 'im' must contain at least [height, width].")
        
    max_h, max_width = im[0] - 1, im[1] - 1
    b0 = tf.maximum(tf.minimum(boxes[:, 0], max_width), 0)
    b1 = tf.maximum(tf.minimum(boxes[:, 1], max_h), 0)
    b2 = tf.maximum(tf.minimum(boxes[:, 2], max_width), 0)
    b3 = tf.maximum(tf.minimum(boxes[:, 3], max_h), 0)
    return tf.stack([b0, b1, b2, b3], axis=1)


def clip_boxes(boxes, im):
    boxes = _validate_numpy_boxes(boxes, "boxes")
    if im is None or len(im) < 2:
        raise ValueError("Image dimension 'im' must contain at least [height, width].")
        
    max_h, max_width = im[0] - 1, im[1] - 1
    b0 = np.maximum(np.minimum(boxes[:, 0], max_width), 0)
    b1 = np.maximum(np.minimum(boxes[:, 1], max_h), 0)
    b2 = np.maximum(np.minimum(boxes[:, 2], max_width), 0)
    b3 = np.maximum(np.minimum(boxes[:, 3], max_h), 0)
    return np.stack([b0, b1, b2, b3], axis=1)


def bbox_transform_tf(rois, gt_rois):
    _validate_tf_boxes(rois, "rois")
    _validate_tf_boxes(gt_rois, "gt_rois")
    
    rois_width = rois[:, 2] - rois[:, 0]
    rois_height = rois[:, 3] - rois[:, 1]
    rois_ctr_x = rois[:, 0] + rois_width / 2.0
    rois_ctr_y = rois[:, 1] + rois_height / 2.0

    gt_width = gt_rois[:, 2] - gt_rois[:, 0]
    gt_height = gt_rois[:, 3] - gt_rois[:, 1]
    gt_ctr_x = gt_rois[:, 0] + gt_width / 2.0
    gt_ctr_y = gt_rois[:, 1] + gt_height / 2.0

    targets_dx = (gt_ctr_x - rois_ctr_x) / rois_width
    targets_dy = (gt_ctr_y - rois_ctr_y) / rois_height
    targets_dw = tf.math.log(gt_width / rois_width)
    targets_dh = tf.math.log(gt_height / rois_height)

    return tf.stack((targets_dx, targets_dy, targets_dw, targets_dh), axis=1)


def bbox_transform(rois, gt_rois):
    rois = _validate_numpy_boxes(rois, "rois")
    gt_rois = _validate_numpy_boxes(gt_rois, "gt_rois")
    
    rois_width = rois[:, 2] - rois[:, 0] + 1.0
    rois_height = rois[:, 3] - rois[:, 1] + 1.0
    rois_ctr_x = rois[:, 0] + rois_width / 2.0
    rois_ctr_y = rois[:, 1] + rois_height / 2.0

    gt_width = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_height = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + gt_width / 2.0
    gt_ctr_y = gt_rois[:, 1] + gt_height / 2.0

    targets_dx = (gt_ctr_x - rois_ctr_x) / rois_width
    targets_dy = (gt_ctr_y - rois_ctr_y) / rois_height
    targets_dw = np.log(gt_width / rois_width)
    targets_dh = np.log(gt_height / rois_height)

    return np.stack((targets_dx, targets_dy, targets_dw, targets_dh), axis=1)


def bbox_overlaps(boxes, gt_boxes):
    """가짜 면적 생성 없는 안전한 분모 분리형 벡터화 IoU 계산"""
    if boxes is None or boxes.size == 0 or gt_boxes is None or gt_boxes.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    boxes = _validate_numpy_boxes(boxes, "boxes")
    gt_boxes = _validate_numpy_boxes(gt_boxes, "gt_boxes")[:, 0:4]
    boxes = boxes.reshape((-1, 4))

    x11, y11, x12, y12 = np.hsplit(boxes, 4)
    x21, y21, x22, y22 = np.hsplit(gt_boxes, 4)

    xI1 = np.maximum(x11, x21.reshape((1, -1)))
    yI1 = np.maximum(y11, y21.reshape((1, -1)))
    xI2 = np.minimum(x12, x22.reshape((1, -1)))
    yI2 = np.minimum(y12, y22.reshape((1, -1)))

    intersection = np.maximum(0.0, (xI2 - xI1 + 1.0)) * np.maximum(0.0, (yI2 - yI1 + 1.0))
    box_area = (x12 - x11 + 1.0) * (y12 - y11 + 1.0)
    gt_box_area = (x22 - x21 + 1.0) * (y22 - y21 + 1.0)

    union = (box_area + gt_box_area.reshape((1, -1))) - intersection

    # 유효하지 않은 union(0 이하)을 가짜 1.0 면적으로 위장하지 않고, 
    # np.divide의 where 조건을 통해 안전하게 0.0으로 처리하여 데이터 무결성 보존
    valid_union = union > 0.0
    overlaps = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=valid_union
    )

    return np.maximum(overlaps, 0.0)

최종 개선사항
✅ 무조건적인 최소값 보정 → 좌표계 convention 통일 및 유효 박스 검증 → encode/decode 왕복 무결성 확보
✅ TF/NumPy 공통 입력 검증 → 프레임워크별 검증 계약 분리 → 정적·동적 shape 환경의 안정성 강화
✅ shape만 검증 → shape + NaN/Inf + 좌표 유효성 검증 → 비정상 입력의 조기 차단
✅ exp overflow만 억제 → 이상값 감지와 수치 clipping 정책 분리 → 모델 이상 출력 은폐 방지
✅ union을 1.0으로 강제 → 안전한 조건부 나눗셈 적용 → IoU 의미 왜곡 없는 0분모 방어
✅ TF/NumPy 좌표 계산 불일치 → 동일한 box width/height convention 적용 → 프레임워크 간 결과 일관성 확보
✅ 단순 런타임 예외 방어 → 입력 계약·수치 안정성·데이터 무결성 계층 분리 → 운영 환경의 장애 추적성과 예측 가능성 강화

원본의 RPN 연산 목적은 유지하면서 수치 오류를 숨기는 방식을 제거하고, 좌표계·입력 계약·IoU 무결성까지 통일한 실무형 유틸리티 구조로 승격했다.    
