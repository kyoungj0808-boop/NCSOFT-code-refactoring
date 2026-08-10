원본코드
import numpy as np

def gpu_nms(polys, thres=0.3, K=100, precision=10000):
    from .nms_kernel import nms as nms_impl
    if len(polys) == 0:
        return np.array([], dtype='float32')
    p = polys.copy()
    #p[:,:8] *= precision
    ret = np.array(nms_impl(p, thres), dtype='int32')
    #ret[:,:8] /= precision
    return ret

def rbox_gpu_nms(polys, thres=0.3, K=100, precision=10000):
    from .nms_kernel import rbox_nms as nms_impl
    if len(polys) == 0:
        return np.array([], dtype='float32')
    p = polys.copy()
    #p[:,:8] *= precision
    ret = np.array(nms_impl(p, thres), dtype='int32')
    #ret[:,:8] /= precision
    return ret

def rbox_iou(quadboxes, query_quadboxes):
    from .nms_kernel import rbox_overlap as nms_impl
    quadboxes_shape = quadboxes.shape
    query_quadboxes_shape = query_quadboxes.shape

    iou_matrix = np.zeros([quadboxes_shape[0], quadboxes_shape[1], query_quadboxes_shape[1]])
    if quadboxes_shape[0] == 0 or query_quadboxes_shape[0] == 0 or query_quadboxes_shape[1] == 0 or quadboxes_shape[1] == 0 :
        print("erro!")
        return iou_matrix

    for i, (each_quad, each_query) in enumerate(zip(quadboxes, query_quadboxes)):
        flatt_iou, idx = nms_impl(each_quad, each_query)
        each_iou = iou_matrix[i]
        each_iou_flatten = np.reshape(each_iou, [-1])
        each_iou_flatten[idx] = flatt_iou
        iou_matrix[i] = np.reshape(each_iou_flatten, [quadboxes_shape[1], query_quadboxes_shape[1]])

        #iou_matrix.append(each_iou)#np.reshape(,[query_quadboxes[1], query_quadboxes_shape[1]]))
    print("rbox iou end")
    return iou_matrix.astype(np.float32)

GPU 커널 연동과 핵심 연산 구조는 효율적이지만, 입력 shape 불일치와 zip()에 의한 데이터 누락을 검증하지 않아 계산 실패를 정상적인 IoU 결과로 위장할 수 있는 것이 가장 치명적인 약점이다.

제안패치
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np


logger = logging.getLogger(__name__)


def _validate_nms_input(
    polys: np.ndarray,
    thres: float,
) -> None:
    """Validate common NMS inputs before entering the GPU kernel."""

    if not isinstance(polys, np.ndarray):
        raise TypeError("polys must be a numpy ndarray.")

    if polys.ndim != 2:
        raise ValueError(
            f"polys must be a 2D array, got shape {polys.shape}."
        )

    if not isinstance(thres, (int, float, np.integer, np.floating)):
        raise TypeError("thres must be a numeric value.")

    if not np.isfinite(thres):
        raise ValueError("thres must be finite.")

    if not 0.0 <= float(thres) <= 1.0:
        raise ValueError("thres must be between 0.0 and 1.0.")

    if polys.size > 0 and not np.all(np.isfinite(polys)):
        raise ValueError("polys must contain only finite values.")


def _validate_nms_result(
    result,
    input_size: int,
) -> np.ndarray:
    """Validate and normalize indices returned by the GPU kernel."""

    result = np.asarray(result)

    if result.ndim != 1:
        raise RuntimeError(
            f"NMS kernel returned invalid shape: {result.shape}."
        )

    if result.size == 0:
        return np.empty(0, dtype=np.int32)

    if not np.issubdtype(result.dtype, np.integer):
        if not np.all(np.isfinite(result)):
            raise RuntimeError(
                "NMS kernel returned non-finite indices."
            )

        if not np.all(result == np.floor(result)):
            raise RuntimeError(
                "NMS kernel returned non-integer indices."
            )

    result = result.astype(np.int64, copy=False)

    if np.any(result < 0) or np.any(result >= input_size):
        raise IndexError(
            "NMS kernel returned an index outside the input range."
        )

    return result.astype(np.int32, copy=False)


def gpu_nms(
    polys: np.ndarray,
    thres: float = 0.3,
    K: int = 100,
) -> np.ndarray:
    """Run GPU-accelerated standard NMS safely."""

    from .nms_kernel import nms as nms_impl

    _validate_nms_input(polys, thres)

    if K is not None:
        if not isinstance(K, (int, np.integer)):
            raise TypeError("K must be an integer or None.")

        if K <= 0:
            raise ValueError("K must be greater than zero.")

    if polys.shape[0] == 0:
        return np.empty(0, dtype=np.int32)

    result = nms_impl(polys.copy(), float(thres))
    indices = _validate_nms_result(result, polys.shape[0])

    if K is not None:
        indices = indices[:K]

    return indices


def rbox_gpu_nms(
    polys: np.ndarray,
    thres: float = 0.3,
    K: int = 100,
) -> np.ndarray:
    """Run GPU-accelerated rotated-box NMS safely."""

    from .nms_kernel import rbox_nms as nms_impl

    _validate_nms_input(polys, thres)

    if K is not None:
        if not isinstance(K, (int, np.integer)):
            raise TypeError("K must be an integer or None.")

        if K <= 0:
            raise ValueError("K must be greater than zero.")

    if polys.shape[0] == 0:
        return np.empty(0, dtype=np.int32)

    result = nms_impl(polys.copy(), float(thres))
    indices = _validate_nms_result(result, polys.shape[0])

    if K is not None:
        indices = indices[:K]

    return indices


def _validate_rbox_input(
    boxes: np.ndarray,
    name: str,
) -> None:
    """Validate batched rotated-box input."""

    if not isinstance(boxes, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray.")

    if boxes.ndim != 3:
        raise ValueError(
            f"{name} must be a 3D array, got shape {boxes.shape}."
        )

    if boxes.shape[2] != 8:
        raise ValueError(
            f"{name} must have 8 coordinates per box, "
            f"got {boxes.shape[2]}."
        )

    if boxes.size > 0 and not np.all(np.isfinite(boxes)):
        raise ValueError(
            f"{name} must contain only finite values."
        )


def rbox_iou(
    quadboxes: np.ndarray,
    query_quadboxes: np.ndarray,
) -> np.ndarray:
    """Compute batched rotated-box IoU matrices safely."""

    from .nms_kernel import rbox_overlap as nms_impl

    _validate_rbox_input(quadboxes, "quadboxes")
    _validate_rbox_input(query_quadboxes, "query_quadboxes")

    batch_size, num_quads, _ = quadboxes.shape
    query_batch_size, num_queries, _ = query_quadboxes.shape

    if batch_size != query_batch_size:
        raise ValueError(
            f"Batch size mismatch: "
            f"{batch_size} != {query_batch_size}."
        )

    iou_matrix = np.zeros(
        (batch_size, num_quads, num_queries),
        dtype=np.float32,
    )

    if num_quads == 0 or num_queries == 0:
        logger.warning(
            "Empty rotated-box dimension detected: "
            "quadboxes=%s, query_quadboxes=%s",
            quadboxes.shape,
            query_quadboxes.shape,
        )
        return iou_matrix

    for batch_idx in range(batch_size):
        flatt_iou, idx = nms_impl(
            quadboxes[batch_idx],
            query_quadboxes[batch_idx],
        )

        flatt_iou = np.asarray(flatt_iou)
        idx = np.asarray(idx)

        if flatt_iou.ndim != 1 or idx.ndim != 1:
            raise RuntimeError(
                "rbox_overlap returned non-1D outputs."
            )

        if flatt_iou.size != idx.size:
            raise RuntimeError(
                "rbox_overlap returned mismatched IoU/index sizes."
            )

        if flatt_iou.size == 0:
            continue

        if not np.all(np.isfinite(flatt_iou)):
            raise RuntimeError(
                "rbox_overlap returned non-finite IoU values."
            )

        idx = idx.astype(np.int64, copy=False)

        flat_size = num_quads * num_queries

        if np.any(idx < 0) or np.any(idx >= flat_size):
            raise IndexError(
                "rbox_overlap returned an index outside "
                "the IoU matrix range."
            )

        iou_matrix[batch_idx].reshape(-1)[idx] = flatt_iou

    logger.debug(
        "rbox_iou completed: batch=%d, boxes=%d, queries=%d",
        batch_size,
        num_quads,
        num_queries,
    )

    return iou_matrix

최종개선사항
✅ zip() 기반 Batch 처리 → Batch 크기 선검증 후 명시적 인덱스 순회 → 입력 개수 불일치에 따른 조용한 IoU 데이터 누락 차단
✅ NMS threshold 무검증 → 유한값·0~1 범위 검증 → 잘못된 threshold의 CUDA 커널 전파 방지
✅ 입력 shape만 검사 → shape와 NaN·Inf까지 함께 검증 → 비정상 좌표가 GPU 연산으로 침투하는 수치 오염 차단
✅ GPU 커널 반환값 즉시 int32 변환 → shape·정수성·유효 index 범위 검증 후 정규화 → 커널 오류의 후속 데이터 손상 전파 방지
✅ rbox_overlap의 IoU와 index 관계 무검증 → 두 배열의 1차원 구조·크기 일치·유한값·index 범위 검증 → 잘못된 IoU 매핑과 메모리 접근 위험 차단
✅ 선언만 되어 있던 K 인자 → 실제 결과 제한 또는 커널 계약에 맞춘 명시적 처리 → API와 실제 동작 사이의 의미 불일치 제거
✅ print() 기반 상태 출력 → 모듈 logger 기반 경고·debug 출력 → 학습 및 추론 파이프라인의 stdout 오염 방지

원본의 GPU NMS·Rotated IoU 연산 구조는 그대로 유지하면서 zip() 데이터 누락·잘못된 threshold·NaN/Inf 입력·커널 반환값 오염까지 방어해, GPU 커널 앞뒤의 입력·출력 계약이 명확한 안정형 NMS wrapper로 승격되었다.    
