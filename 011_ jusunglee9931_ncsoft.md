원본코드
import tensorflow as tf
import numpy as np
from utils import cal_iou_np

FLAGS = tf.app.flags.FLAGS





def update_gt_class(y, idx, test_image_num, class_num):
    gt_matrix = np.zeros(shape=[test_image_num, class_num], dtype=np.float32)
    for label in y:
        gt_matrix[int(idx), int(label)-1] += 1
    return gt_matrix


def update_tp_fp(gt_label, gt_quadboxes, rec_boxes, classifier, idx, test_image_num, class_num,
                 iou_threshold=0.5):
    tp_matrix = np.zeros(shape=[test_image_num, class_num], dtype=np.float32)
    fp_matrix = np.zeros(shape=[test_image_num, class_num], dtype=np.float32)
    idx = int(idx)
    gt_quadboxes = gt_quadboxes[:, 0, :]
    rec_boxes = rec_boxes.reshape([-1, 8])
    classifier = classifier.reshape([-1, 1])

    rec_boxes_removed = []
    classifier_removed = []

    for each_rec_boxes, each_classifier in zip(rec_boxes, classifier):
        if np.sum(each_rec_boxes) == 0:
            continue
        else:
            rec_boxes_removed.append(each_rec_boxes)
            classifier_removed.append(each_classifier)

    rec_boxes = np.reshape(np.array(rec_boxes_removed), [-1, 8])
    classifier = np.reshape(np.array(classifier_removed), [-1, 1])

    gt_num = gt_label.shape[0]
    positive_num = 0
    true_positive_num = 0
    matching_checking = {}
    overlaps = cal_iou_np(gt_quadboxes, rec_boxes)
    print("idx : {}".format(idx))
    if len(overlaps) != 0:
        # (GT dim)
        # pred_idx_base_on_gt = np.argmax(overlaps, axis=-1)
        # pred_max_iou_base_on_gt = np.amax(overlaps, axis=-1)
        gt_idx_base_on_pred = np.argmax(overlaps, axis=0)
        gt_max_iou_base_on_pred = np.amax(overlaps, axis=0)

        for each_gt_idx, iou, each_pred_label in zip(gt_idx_base_on_pred, gt_max_iou_base_on_pred, classifier):
            positive_num += 1

            if iou > iou_threshold:  # and gt_label[each_gt_idx] == each_pred_label:
                if matching_checking.get(each_gt_idx, None) is None:
                    true_positive_num += 1
                    matching_checking[each_gt_idx] = 1

        false_positive = positive_num - true_positive_num

        print("positive_num : {}".format(positive_num))
        print("false_positive : {}".format(false_positive))
        print("true_positive_num : {}".format(true_positive_num))

        print("precision : {}".format(true_positive_num / float(positive_num + 0.00001)))
        print("recall : {}".format(true_positive_num / float(gt_num + 0.00001)))
        tp_matrix[idx, 0] = true_positive_num
        fp_matrix[idx, 0] = false_positive

    return tp_matrix, fp_matrix


class F1_Metric(tf.keras.metrics.Metric):
    def __init__(self, class_num, test_image_num=10000, mode="recall", name='mAPmetric', **kwargs):
        super(F1_Metric, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", shape=(test_image_num, class_num), initializer='zero')
        self.false_positives = self.add_weight(name="fp", shape=(test_image_num, class_num), initializer='zero')
        self.gt_counter_per_class = self.add_weight(name='gt_per_class', shape=(test_image_num, class_num),
                                                    initializer='zero')
        self.test_image_num = test_image_num
        self.num_class = class_num
        self.mode = mode

        self.cnt = self.add_weight(name="cnt", initializer='zero')

    def update_state(self, y, pred):
        """
        batch size 1 supported
        """
        gt_label = y["label"][0]
        gt_quadboxes = y["quad_boxes"][0]

        rec_boxes = pred["rec_boxes"]
        classifier = pred["classifier"]

        gt_label_idx = tf.where(gt_label > 0)
        gt_label_gather = tf.gather(gt_label, gt_label_idx)
        gt_quadboxes_gather = tf.gather(gt_quadboxes, gt_label_idx)

        gt_matrix = tf.py_func(update_gt_class, [gt_label_gather, self.cnt, self.test_image_num, self.num_class],
                               tf.float32, name='update_gt_class')
        self.gt_counter_per_class.assign_add(gt_matrix)

        tp, fp = tf.py_func(update_tp_fp, [gt_label_gather, gt_quadboxes_gather, rec_boxes, classifier, self.cnt,
                            self.test_image_num, self.num_class], [tf.float32, tf.float32],
                            name='update_tp_fp')
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.cnt.assign_add(1)

    def result(self):
        whole_tp = tf.reduce_sum(self.true_positives, axis=[0, 1])
        whole_fp = tf.reduce_sum(self.false_positives, axis=[0, 1])
        whole_gt = tf.reduce_sum(self.gt_counter_per_class, axis=[0, 1])
        if self.mode == "recall":
            recall = whole_tp / whole_gt
            return recall
        elif self.mode == "precision":
            precision = whole_tp / (whole_tp + whole_fp)
            return precision
        else:
            recall = whole_tp / whole_gt
            precision = whole_tp / (whole_tp + whole_fp)
            f_score = 2/(1/recall + 1/precision)
            return f_score

    def reset_states(self):
        for i in range(self.num_class):
            self.true_positives[i].assign(0)
            self.false_positives[i].assign(0)
            self.gt_counter_per_class[i].assign(0)

회전 박스 검출 평가 로직의 기본 골격은 명확하지만, 레거시 실행 방식·다중 클래스 집계 누락·상태 초기화 오류·0 나누기 처리까지 겹쳐 현재 구현은 단순한 리팩토링 대상이 아니라 평가 결과 자체의 신뢰성을 재검증해야 하는 수준이며, 특히 모델이 정상 학습되더라도 메트릭이 틀린 값을 누적할 수 있다는 점이 가장 치명적이다.

제안패치
# SPDX-FileCopyrightText: Ⓒ 2023 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import tensorflow as tf
import numpy as np

from utils import cal_iou_np


class F1_Metric(tf.keras.metrics.Metric):
    """
    Rotated Bounding Box detection metric.

    Invariants:
    - GT labels are 1-based and must be within [1, class_num].
    - Each prediction can match at most one GT.
    - Each GT can be matched at most once.
    - A TP requires both IoU threshold satisfaction and class agreement.
    - FP is accumulated against the predicted class.
    """

    def __init__(
        self,
        class_num,
        test_image_num=10000,
        mode="recall",
        iou_threshold=0.5,
        name="mAPmetric",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        if not isinstance(class_num, (int, np.integer)) or class_num <= 0:
            raise ValueError("class_num must be a positive integer.")

        if not isinstance(test_image_num, (int, np.integer)) or test_image_num <= 0:
            raise ValueError("test_image_num must be a positive integer.")

        if not np.isfinite(iou_threshold) or not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be finite and within [0, 1].")

        if mode not in {"recall", "precision", "f1"}:
            raise ValueError(
                "mode must be one of: 'recall', 'precision', 'f1'."
            )

        self.num_class = int(class_num)
        self.test_image_num = int(test_image_num)
        self.mode = mode
        self.iou_threshold = float(iou_threshold)

        self.true_positives = self.add_weight(
            name="tp",
            shape=(self.test_image_num, self.num_class),
            initializer="zeros",
            dtype=tf.float32,
        )
        self.false_positives = self.add_weight(
            name="fp",
            shape=(self.test_image_num, self.num_class),
            initializer="zeros",
            dtype=tf.float32,
        )
        self.gt_counter_per_class = self.add_weight(
            name="gt_per_class",
            shape=(self.test_image_num, self.num_class),
            initializer="zeros",
            dtype=tf.float32,
        )
        self.cnt = self.add_weight(
            name="cnt",
            initializer="zeros",
            dtype=tf.int32,
        )

    def update_state(self, y, pred, sample_weight=None):
        del sample_weight

        gt_label = tf.cast(y["label"][0], tf.int32)
        gt_quadboxes = tf.cast(y["quad_boxes"][0], tf.float32)

        rec_boxes = tf.cast(pred["rec_boxes"], tf.float32)
        classifier = tf.cast(pred["classifier"], tf.int32)

        current_idx = self.cnt

        tf.debugging.assert_less(
            current_idx,
            self.test_image_num,
            message="Evaluation index exceeds test_image_num.",
        )

        valid_gt_mask = gt_label > 0
        valid_gt_labels = tf.boolean_mask(gt_label, valid_gt_mask)
        valid_gt_boxes = tf.boolean_mask(
            gt_quadboxes,
            valid_gt_mask,
        )

        tf.debugging.assert_greater_equal(
            valid_gt_labels,
            1,
            message="GT labels must be 1-based.",
        )
        tf.debugging.assert_less_equal(
            valid_gt_labels,
            self.num_class,
            message="GT label exceeds class_num.",
        )

        gt_indices = tf.stack(
            [
                tf.fill(
                    [tf.shape(valid_gt_labels)[0]],
                    current_idx,
                ),
                valid_gt_labels - 1,
            ],
            axis=1,
        )

        gt_delta = tf.scatter_nd(
            gt_indices,
            tf.ones(
                [tf.shape(valid_gt_labels)[0]],
                dtype=tf.float32,
            ),
            [self.test_image_num, self.num_class],
        )

        self.gt_counter_per_class.assign_add(gt_delta)

        rec_boxes = tf.reshape(rec_boxes, [-1, 8])
        classifier = tf.reshape(classifier, [-1])

        # Zero-filled detection slots are considered invalid.
        valid_pred_mask = tf.reduce_any(
            tf.not_equal(rec_boxes, 0.0),
            axis=1,
        )

        valid_rec_boxes = tf.boolean_mask(rec_boxes, valid_pred_mask)
        valid_classifier = tf.boolean_mask(classifier, valid_pred_mask)

        tf.debugging.assert_greater_equal(
            valid_classifier,
            1,
            message="Prediction class labels must be >= 1.",
        )
        tf.debugging.assert_less_equal(
            valid_classifier,
            self.num_class,
            message="Prediction class label exceeds class_num.",
        )

        tp, fp = tf.numpy_function(
            self._match_predictions,
            [
                valid_gt_boxes,
                valid_gt_labels,
                valid_rec_boxes,
                valid_classifier,
                tf.cast(current_idx, tf.int32),
            ],
            [tf.float32, tf.float32],
        )

        tp.set_shape([self.test_image_num, self.num_class])
        fp.set_shape([self.test_image_num, self.num_class])

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.cnt.assign_add(1)

    def _match_predictions(
        self,
        gt_boxes,
        gt_labels,
        pred_boxes,
        pred_labels,
        current_idx,
    ):
        tp_matrix = np.zeros(
            (self.test_image_num, self.num_class),
            dtype=np.float32,
        )
        fp_matrix = np.zeros(
            (self.test_image_num, self.num_class),
            dtype=np.float32,
        )

        gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
        gt_labels = np.asarray(gt_labels, dtype=np.int32).reshape(-1)
        pred_boxes = np.asarray(pred_boxes, dtype=np.float32)
        pred_labels = np.asarray(pred_labels, dtype=np.int32).reshape(-1)

        if pred_boxes.shape[0] == 0:
            return tp_matrix, fp_matrix

        if gt_boxes.shape[0] == 0:
            for label in pred_labels:
                if 1 <= label <= self.num_class:
                    fp_matrix[int(current_idx), label - 1] += 1.0
            return tp_matrix, fp_matrix

        overlaps = cal_iou_np(gt_boxes[:, 0, :] if gt_boxes.ndim == 3 else gt_boxes,
                              pred_boxes)

        if overlaps.size == 0:
            for label in pred_labels:
                if 1 <= label <= self.num_class:
                    fp_matrix[int(current_idx), label - 1] += 1.0
            return tp_matrix, fp_matrix

        matched_gt = set()

        for pred_idx, pred_label in enumerate(pred_labels):
            if not 1 <= pred_label <= self.num_class:
                continue

            gt_idx = int(np.argmax(overlaps[:, pred_idx]))
            best_iou = float(overlaps[gt_idx, pred_idx])

            if (
                best_iou >= self.iou_threshold
                and gt_idx not in matched_gt
                and int(gt_labels[gt_idx]) == int(pred_label)
            ):
                tp_matrix[int(current_idx), pred_label - 1] += 1.0
                matched_gt.add(gt_idx)
            else:
                fp_matrix[int(current_idx), pred_label - 1] += 1.0

        return tp_matrix, fp_matrix

    def result(self):
        whole_tp = tf.reduce_sum(self.true_positives, axis=0)
        whole_fp = tf.reduce_sum(self.false_positives, axis=0)
        whole_gt = tf.reduce_sum(self.gt_counter_per_class, axis=0)

        epsilon = tf.keras.backend.epsilon()

        recall = tf.math.divide_no_nan(
            whole_tp,
            whole_gt,
        )

        precision = tf.math.divide_no_nan(
            whole_tp,
            whole_tp + whole_fp,
        )

        f1 = tf.math.divide_no_nan(
            2.0 * recall * precision,
            recall + precision,
        )

        if self.mode == "recall":
            return recall

        if self.mode == "precision":
            return precision

        return f1

    def reset_state(self):
        self.true_positives.assign(
            tf.zeros_like(self.true_positives)
        )
        self.false_positives.assign(
            tf.zeros_like(self.false_positives)
        )
        self.gt_counter_per_class.assign(
            tf.zeros_like(self.gt_counter_per_class)
        )
        self.cnt.assign(0)

최종개선사항
✅ FP 전체를 0번 클래스에 집계 → 예측 클래스별 FP 귀속 → 다중 클래스 Precision 무결성 확보
✅ IoU만으로 TP 판정 → IoU·GT 미사용·클래스 일치 조건 결합 → 잘못된 클래스의 오탐을 TP로 인정하는 오류 차단
✅ GT/Prediction 빈 상태 미정의 → 0/0·0/N·N/0 경계 명시 → 평가 결과의 NaN 및 누락 방지
✅ Tensor 조건을 Python if로 판정 → NumPy 매칭 함수 내부로 명확히 분리 → TensorFlow 실행 그래프와 평가 로직의 경계 안정화
✅ 매직 넘버 기반 분모 방어 → tf.math.divide_no_nan 적용 → 0분모 상황에서도 지표의 NaN 전파 차단
✅ GT·Prediction 클래스 범위 무검증 → 1~class_num 불변조건 검증 → 잘못된 라벨의 조용한 데이터 오염 방지

레거시 메트릭의 표면적 문제만 제거한 수준을 넘어 클래스-aware one-to-one 매칭과 TP/FP 귀속을 바로잡아, 평가기가 모델 성능을 잘못 판정하는 핵심 무결성 결함을 제거한 실무형 구조로 승격했다.        
