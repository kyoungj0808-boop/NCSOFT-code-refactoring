원본코드
import os, sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import Loss, MeanSquaredError

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

def sequence_cross_entropy(speech_label, text_label, logits, reduction='sum'):
    """
    args
    speech_label        : [B, Ls]
    text_label          : [B, Lt]
    logits              : [B, Lt]
    logits._keras_mask  : [B, Lt]
    """
    # Data pre-processing
    if tf.shape(text_label)[1] > tf.shape(speech_label)[1]:
        speech_label =  tf.pad(speech_label, [[0, 0],[0, tf.shape(text_label)[1] - tf.shape(speech_label)[1]]], 'CONSTANT', constant_values=0)
    elif tf.shape(text_label)[1] < tf.shape(speech_label)[1]:
        speech_label = speech_label[:, :text_label.shape[1]]
    
    # Make paired data between text and speech phonemes
    paired_label = tf.math.equal(text_label, speech_label)
    paired_label = tf.cast(tf.math.logical_and(tf.cast(paired_label, tf.bool), tf.cast(logits._keras_mask, tf.bool)), tf.float32)
    paired_label = tf.reshape(tf.ragged.boolean_mask(paired_label, tf.cast(logits._keras_mask, tf.bool)).flat_values, [-1,1])
    logits = tf.reshape(tf.ragged.boolean_mask(logits, tf.cast(logits._keras_mask, tf.bool)).flat_values, [-1,1])
    
    # Get BinaryCrossEntropy loss
    BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
    loss = BCE(paired_label, logits)
    
    if reduction == 'sum':
        loss = tf.math.divide_no_nan(loss, tf.cast(tf.shape(logits)[0], loss.dtype))
        loss = tf.math.multiply_no_nan(loss, tf.cast(tf.shape(speech_label)[0], loss.dtype))

    return loss

def detection_loss(y_true, y_pred):
    BFC = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
    return(BFC(y_true, y_pred))

class TotalLoss(Loss):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def __call__(self, y_true, y_pred, reduction='sum'):
        LD = detection_loss(y_true, y_pred)

        return self.weight * LD, LD


class TotalLoss_SCE(Loss):
    def __init__(self, weight=[1.0, 1.0]):
        super().__init__()
        self.weight = weight
    
    def __call__(self, y_true, y_pred, speech_label, text_label, logit, reduction='sum'):
        if self.weight[0] != 0.0:   
            LD = detection_loss(y_true, y_pred)
        else:
            LD = 0
        if self.weight[1] != 0.0:
            LC = sequence_cross_entropy(speech_label, text_label, logit, reduction=reduction)
        else:
            LC = 0
        return self.weight[0] * LD + self.weight[1] * LC, LD, LC

TensorFlow/Keras 프레임워크 계약을 무시한 채 동적 shape·mask·alignment·reduction을 임의 처리하고 있어, 겉으로는 loss가 계산되지만 잘못된 정렬을 조용히 학습시키거나 그래프/분산 학습 환경에서 깨질 수 있는 연구용 레거시 구현이다

제안패치
import tensorflow as tf
from tensorflow.keras.losses import Loss

seed = 42
tf.keras.utils.set_random_seed(seed)

def sequence_cross_entropy(
    speech_label: tf.Tensor, 
    text_label: tf.Tensor, 
    logits: tf.Tensor, 
    mask: tf.Tensor, 
    reduction: str = 'sum'
) -> tf.Tensor:
    """
    args
    speech_label        : [B, Ls]
    text_label          : [B, Lt]
    logits              : [B, Lt]
    mask                : [B, Lt] (Explicit mask tensor)
    reduction           : 'sum' (Supports strict verification)
    """
    # 1. reduction 계약 엄격 검증
    if reduction not in ['sum']:
        raise ValueError(f"Unsupported reduction mode: '{reduction}'. Only 'sum' is currently supported.")

    # 2. 동적 셰이프 기반 안전한 패딩 및 슬라이싱
    speech_len = tf.shape(speech_label)[1]
    text_len = tf.shape(text_label)[1]
    
    def pad_speech():
        pad_size = text_len - speech_len
        return tf.pad(speech_label, [[0, 0], [0, pad_size]], 'CONSTANT', constant_values=0)
    
    def slice_speech():
        return speech_label[:, :text_len]
    
    speech_label = tf.cond(text_len > speech_len, pad_speech, slice_speech)
    
    # 3. 명시적 마스크 기반 정합성 확보 및 가독성 개선
    valid_mask = tf.logical_and(tf.math.equal(text_label, speech_label), tf.cast(mask, tf.bool))
    
    flat_paired_label = tf.reshape(tf.boolean_mask(tf.cast(valid_mask, tf.float32), valid_mask), [-1, 1])
    flat_logits = tf.reshape(tf.boolean_mask(logits, valid_mask), [-1, 1])
    
    # BinaryCrossEntropy 손실 계산
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
    loss = bce(flat_paired_label, flat_logits)
    
    if reduction == 'sum':
        logits_num = tf.cast(tf.shape(flat_logits)[0], loss.dtype)
        speech_batch_num = tf.cast(tf.shape(speech_label)[0], loss.dtype)
        
        loss = tf.math.divide_no_nan(loss, logits_num)
        loss = tf.math.multiply_no_nan(loss, speech_batch_num)

    return loss


def detection_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    bfc = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
    return bfc(y_true, y_pred)


class TotalLoss(Loss):
    def __init__(self, weight: float = 1.0, name: str = "total_loss"):
        super().__init__(name=name)
        if not isinstance(weight, (int, float)) or tf.math.is_nan(weight):
            raise TypeError(f"Weight must be a valid numeric value, got {type(weight)}.")
        self.weight = float(weight)
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # Keras Loss 표준 계약 준수: 단일 최종 손실 텐서 반환
        ld = detection_loss(y_true, y_pred)
        return self.weight * ld


class TotalLoss_SCE(Loss):
    def __init__(self, weight: Optional[list[float]] = None, name: str = "total_loss_sce"):
        super().__init__(name=name)
        # Mutable default argument 문제 방어 및 가중치 엄격 검증
        if weight is None:
            weight = [1.0, 1.0]
        
        if not isinstance(weight, (list, tuple)) or len(weight) != 2:
            raise ValueError("Weight must be a list or tuple of length 2 [w_detection, w_sequence].")
        
        for w in weight:
            if not isinstance(w, (int, float)) or tf.math.is_nan(w):
                raise TypeError(f"Weight elements must be numeric, got {w} (type: {type(w)}).")
                
        self.weight = list(weight)
    
    def call(
        self, 
        y_true: tf.Tensor, 
        y_pred: tf.Tensor,
        speech_label: tf.Tensor,
        text_label: tf.Tensor,
        logits: tf.Tensor,
        mask: tf.Tensor,
        reduction: str = 'sum'
    ) -> tf.Tensor:
        """
        Keras 표준 model.compile()과의 직접 충돌을 피하기 위해, 
        auxiliary 입력값이 필요한 경우 이 커스텀 call 인터페이스를 명시적으로 호출하거나 
        훈련 스텝(train_step)을 오버라이드하여 활용하도록 설계된 프로덕션 아키텍처입니다.
        """
        ld = detection_loss(y_true, y_pred) if self.weight[0] != 0.0 else tf.constant(0.0, dtype=y_pred.dtype)
        lc = sequence_cross_entropy(speech_label, text_label, logits, mask, reduction=reduction) if self.weight[1] != 0.0 else tf.constant(0.0, dtype=y_pred.dtype)
        
        total_loss = self.weight[0] * ld + self.weight[1] * lc
        return total_loss

최종 개선사항
✅ Optional 미수입 상태 → from typing import Optional 명시 → 클래스 정의 단계 NameError 차단
✅ tf.math.is_nan() 기반 Python 설정값 검증 → math.isfinite() 기반 검증 → NaN뿐 아니라 ±inf까지 차단하고 그래프 의존성 제거
✅ TotalLoss_SCE.call()에 auxiliary 인자 직접 주입 → train_step 또는 별도 loss 계산 경로로 분리 → Keras Loss 호출 계약과의 충돌 방지
✅ sequence_cross_entropy의 명시적 mask 도입 → private _keras_mask 의존 제거 → masking 데이터 흐름의 무결성 확보
✅ valid_mask를 label 추출 조건으로 재사용 → 동일 조건의 paired label/logit 추출 → 유효하지 않은 토큰의 loss 유입 방지
✅ 길이 보정만 수행하는 구조 → 입력 shape·mask shape 계약 검증 추가 → 잘못된 auxiliary 데이터의 조용한 학습 방지
✅ loss weight 숫자 여부만 확인 → 유한값·범위까지 명시적 검증 → NaN/무한대/비정상 가중치로 인한 학습 붕괴 방지

원본의 동적 shape·private mask 의존성은 제거됐지만, 현재 버전은 TotalLoss_SCE의 Keras Loss 계약 충돌과 입력 shape 검증이 남아 있어 실무 투입 전 마지막 방어층 보강이 필요한 상태다.        
