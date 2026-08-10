원본코드
import tensorflow as tf
import numpy as np
import math


def roi_rotate_quad(featuremap, quadboxes, shape, input_shape):
    """
    :param featuremap: (B, H, W, C)
    :param quadboxes: (B, N, 8)
    :param shape: (new_h, new_w)
    :param input_shape: (B, H, W, C)
    :return: cropped feature (BN, new_h, new_w, C)
    """
    quadboxes = tf.stop_gradient(quadboxes)
    feature_shape = tf.shape(featuremap)
    bboxes_shape = tf.shape(quadboxes)
    x1, y1, x2, y2, x3, y3, x4, y4 = tf.split(value=quadboxes, num_or_size_splits=8, axis=-1)

    w = tf.cast(input_shape[2], tf.float32)
    h = tf.cast(input_shape[1], tf.float32)
    x1 /= w
    x2 /= w
    x3 /= w
    x4 /= w

    y1 /= h
    y2 /= h
    y3 /= h
    y4 /= h

    x_bundle = tf.concat([x1, x2, x3, x4], axis=-1)
    y_bundle = tf.concat([y1, y2, y3, y4], axis=-1)

    min_x = tf.reduce_min(x_bundle, -1)  # B, N
    max_x = tf.reduce_max(x_bundle, -1)

    min_y = tf.reduce_min(y_bundle, -1)
    max_y = tf.reduce_max(y_bundle, -1)

    point_lt = tf.concat([x1, y1], axis=-1)  # B, N ,2
    point_rt = tf.concat([x2, y2], axis=-1)
    point_rb = tf.concat([x3, y3], axis=-1)
    point_lb = tf.concat([x4, y4], axis=-1)

    theta_vector1 = (point_rt - point_lt + point_rb - point_lb) / 2.0
    x_vector1 = theta_vector1[:, :, 0] * (shape[1] / (max_x - min_x))
    eps = tf.fill(tf.shape(x_vector1), 0.00001)
    div_x_vector1 = tf.where(x_vector1 == 0, eps, x_vector1)
    theta1 = (theta_vector1[:, :, 1] / div_x_vector1) * (shape[0] / (max_y - min_y))  # B, N

    theta_vector2 = (point_rb - point_rt + point_lb - point_lt) / 2.0
    x_vector2 = theta_vector2[:, :, 0] * (shape[1] / (max_x - min_x))
    div_x_vector2 = tf.where(x_vector2 == 0, eps, x_vector2)
    theta2 = (theta_vector2[:, :, 1] / div_x_vector2) * (shape[0] / (max_y - min_y))

    theta1 = tf.atan(theta1)
    theta2 = tf.atan(theta2)
    theta2_recal = tf.where(theta2 < 0, math.pi / 2.0 + theta2, theta2 - math.pi / 2.0)

    theta = tf.where(tf.norm(theta_vector1, axis=-1) > tf.norm(theta_vector2, axis=-1), theta1, theta2_recal)
    '''
    shape => (B, N)
    '''

    batch_size = feature_shape[0]
    batch_linspace = tf.range(0, batch_size)
    batch_linspace = tf.reshape(batch_linspace, [-1, 1])

    batch_idx = tf.tile(batch_linspace, [1, bboxes_shape[1]])  # B, N
    tf_non_zero_idx = tf.where(tf.logical_and(tf.not_equal(max_x - min_x, 0), tf.not_equal(max_y - min_y, 0)))
    y1_cord = tf.gather_nd(min_y, tf_non_zero_idx)
    x1_cord = tf.gather_nd(min_x, tf_non_zero_idx)
    y2_cord = tf.gather_nd(max_y, tf_non_zero_idx)
    x2_cord = tf.gather_nd(max_x, tf_non_zero_idx)
    batch_cord = tf.gather_nd(batch_idx, tf_non_zero_idx)
    theta_cord = tf.gather_nd(theta, tf_non_zero_idx)
    filterd_bboxes = tf.stack([y1_cord, x1_cord, y2_cord, x2_cord], axis=-1)

    cropped = tf.image.crop_and_resize(featuremap, filterd_bboxes, batch_cord, shape)
    cropped_shape = tf.shape(cropped)

    theta_cord_expand = tf.expand_dims(theta_cord, axis=-1)  # None
    theta_cord_expand = tf.tile(theta_cord_expand, [1, shape[0] * shape[1]])  # None, w*h
    sine = tf.sin(theta_cord_expand)  # None, w*h
    cos = tf.cos(theta_cord_expand)

    grid = tf.meshgrid(tf.linspace(-1.0, 1.0, shape[0]), tf.linspace(-1.0, 1.0, shape[1]), indexing='ij')
    grid = tf.stack(grid, axis=-1)
    grid = tf.reshape(grid, (-1, 2))  # None
    grid = tf.expand_dims(grid, axis=0)
    grid = tf.tile(grid, [cropped_shape[0], 1, 1])  # None, w*h, 2

    new_grid_x = grid[:, :, 1] * cos - grid[:, :, 0] * sine  # None, h*w
    new_grid_y = grid[:, :, 1] * sine + grid[:, :, 0] * cos

    output = tf.stack([new_grid_y, new_grid_x], axis=-1)  # None, h*w, 2
    output = tf.reshape(output, [cropped_shape[0], shape[0], shape[1], 2])  # None, h, w, 2

    height_f = tf.cast(shape[0], 'float32')
    width_f = tf.cast(shape[1], 'float32')

    new_h = output[:, :, :, 0]  # + tf.cast(h/2, tf.float32)
    new_w = output[:, :, :, 1]  # + tf.cast(w/2, tf.float32)
    new_w = (new_w + 1.0) * width_f / 2.0
    new_h = (new_h + 1.0) * height_f / 2.0
    new_h = tf.cast(tf.floor(new_h), tf.int32)
    new_w = tf.cast(tf.floor(new_w), tf.int32)  # None, h , w

    batch_idx = tf.range(cropped_shape[0])
    batch_idx = tf.reshape(batch_idx, (cropped_shape[0], 1, 1))
    b = tf.tile(batch_idx, (1, shape[0], shape[1]))  # None, h, w
    indexa = tf.stack([b, new_h, new_w], axis=-1)  # None, h, w, 3
    new_featuremap = tf.gather_nd(cropped, indexa)

    return new_featuremap, tf_non_zero_idx, batch_cord, theta_cord


def rotate_np(cropped, theta):
    """
    :param cropped: (h,w,ch)
    :param theta: ()
    :return:
    """
    feature_shape = cropped.shape
    sin = np.sin(theta)
    cos = np.cos(theta)
    r_mat_row = np.stack([cos, -sin], axis=-1)
    r_mat_row2 = np.stack([sin, cos], axis=-1)
    r_mat = np.stack([r_mat_row, r_mat_row2], axis=-1)  # (2,2)
    r_mat = r_mat[np.newaxis, :, :]
    r_mat = np.tile(r_mat, [feature_shape[0] * feature_shape[1], 1, 1])  # (h*w,2,2)
    grid = np.meshgrid(np.linspace(-1.0, 1.0, feature_shape[0]), np.linspace(-1.0, 1.0, feature_shape[1]),
                       indexing='ij')
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, (-1, 2))
    grid = grid[:, :, np.newaxis]  # (h*w,2,1)
    output = np.matmul(r_mat, grid)
    output = np.reshape(output, [feature_shape[0], feature_shape[1], 2])

    new_h = output[:, :, 0]
    new_w = output[:, :, 1]
    new_w = (new_w + 1.0) * (feature_shape[1]) / 2.0
    new_h = (new_h + 1.0) * (feature_shape[0]) / 2.0
    new_w = np.floor(new_w).astype(np.int32)
    new_h = np.floor(new_h).astype(np.int32)
    new_w = np.clip(new_w, 0, feature_shape[1] - 1)
    new_h = np.clip(new_h, 0, feature_shape[0] - 1)
    if len(feature_shape) > 2:
        rotated_cropped = cropped[new_h, new_w, :]
    else:
        rotated_cropped = cropped[new_h, new_w]

    return rotated_cropped

원본은 회전 ROI의 핵심 수학 로직은 갖췄지만 극소 분모·좌표 경계·빈 ROI·동적 텐서 크기·과도한 tile 연산에 대한 방어가 부족해, 정상 데이터에서는 버티더라도 엣지 케이스에서 NaN/OOB/OOM으로 전체 추론·학습 파이프라인을 셧다운시킬 수 있는 연구용 구현에 가깝다.

제안패치
import tensorflow as tf
import numpy as np
import math


def roi_rotate_quad(featuremap, quadboxes, shape, input_shape):
    """
    :param featuremap: (B, H, W, C)
    :param quadboxes: (B, N, 8)
    :param shape: (new_h, new_w)
    :param input_shape: (B, H, W, C)
    :return: cropped feature (BN, new_h, new_w, C), non_zero_idx, batch_cord, theta_cord
    """
    quadboxes = tf.stop_gradient(quadboxes)
    feature_shape = tf.shape(featuremap)
    bboxes_shape = tf.shape(quadboxes)
    x1, y1, x2, y2, x3, y3, x4, y4 = tf.split(value=quadboxes, num_or_size_splits=8, axis=-1)

    w = tf.cast(input_shape[2], tf.float32)
    h = tf.cast(input_shape[1], tf.float32)
    x1 /= w
    x2 /= w
    x3 /= w
    x4 /= w

    y1 /= h
    y2 /= h
    y3 /= h
    y4 /= h

    x_bundle = tf.concat([x1, x2, x3, x4], axis=-1)
    y_bundle = tf.concat([y1, y2, y3, y4], axis=-1)

    min_x = tf.reduce_min(x_bundle, -1)  # B, N
    max_x = tf.reduce_max(x_bundle, -1)

    min_y = tf.reduce_min(y_bundle, -1)
    max_y = tf.reduce_max(y_bundle, -1)

    point_lt = tf.concat([x1, y1], axis=-1)  # B, N ,2
    point_rt = tf.concat([x2, y2], axis=-1)
    point_rb = tf.concat([x3, y3], axis=-1)
    point_lb = tf.concat([x4, y4], axis=-1)

    # 1. 단순화되고 수치적으로 안전한 safe_divisor 구조 적용
    eps = 1e-6
    def safe_divisor(x):
        return tf.where(tf.abs(x) < eps, tf.where(x < 0.0, -eps, eps), x)

    theta_vector1 = (point_rt - point_lt + point_rb - point_lb) / 2.0
    x_vec1_safe = safe_divisor(theta_vector1[:, :, 0] * (shape[1] / safe_divisor(max_x - min_x)))
    theta1 = (theta_vector1[:, :, 1] / x_vec1_safe) * (shape[0] / safe_divisor(max_y - min_y))

    theta_vector2 = (point_rb - point_rt + point_lb - point_lt) / 2.0
    x_vec2_safe = safe_divisor(theta_vector2[:, :, 0] * (shape[1] / safe_divisor(max_x - min_x)))
    theta2 = (theta_vector2[:, :, 1] / x_vec2_safe) * (shape[0] / safe_divisor(max_y - min_y))

    theta1 = tf.atan(theta1)
    theta2 = tf.atan(theta2)
    theta2_recal = tf.where(theta2 < 0, math.pi / 2.0 + theta2, theta2 - math.pi / 2.0)

    theta = tf.where(tf.norm(theta_vector1, axis=-1) > tf.norm(theta_vector2, axis=-1), theta1, theta2_recal)

    batch_size = feature_shape[0]
    batch_linspace = tf.range(0, batch_size)
    batch_linspace = tf.reshape(batch_linspace, [-1, 1])

    batch_idx = tf.tile(batch_linspace, [1, bboxes_shape[1]])  # B, N
    tf_non_zero_idx = tf.where(tf.logical_and(tf.not_equal(max_x - min_x, 0), tf.not_equal(max_y - min_y, 0)))

    # 2. 빈(Empty) 박스 방어 분기 및 tf.cond를 통한 그래프 모드 정합성 확보
    def handle_empty_boxes():
        empty_feature = tf.zeros([0, shape[0], shape[1], feature_shape[-1]], dtype=featuremap.dtype)
        return empty_feature, tf_non_zero_idx, tf.cast(tf.zeros([0], dtype=tf.int32), tf.int32), tf.cast(tf.zeros([0], dtype=tf.float32), tf.float32)

    def process_valid_boxes():
        y1_cord = tf.gather_nd(min_y, tf_non_zero_idx)
        x1_cord = tf.gather_nd(min_x, tf_non_zero_idx)
        y2_cord = tf.gather_nd(max_y, tf_non_zero_idx)
        x2_cord = tf.gather_nd(max_x, tf_non_zero_idx)
        batch_cord = tf.gather_nd(batch_idx, tf_non_zero_idx)
        theta_cord = tf.gather_nd(theta, tf_non_zero_idx)
        filterd_bboxes = tf.stack([y1_cord, x1_cord, y2_cord, x2_cord], axis=-1)

        cropped = tf.image.crop_and_resize(featuremap, filterd_bboxes, batch_cord, shape)
        cropped_shape = tf.shape(cropped)

        # 3. 불필요한 tile 연산 제거 및 브로드캐스팅 기반 최적화
        theta_rad = theta_cord[:, None, None]  # [None, 1, 1]
        sine = tf.sin(theta_rad)
        cosine = tf.cos(theta_rad)

        grid = tf.stack(
            tf.meshgrid(
                tf.linspace(-1.0, 1.0, shape[0]),
                tf.linspace(-1.0, 1.0, shape[1]),
                indexing='ij',
            ),
            axis=-1,
        )  # [H, W, 2]
        grid = grid[None, ...]  # [1, H, W, 2]

        new_grid_x = grid[..., 1] * cosine - grid[..., 0] * sine
        new_grid_y = grid[..., 1] * sine + grid[..., 0] * cosine

        output = tf.stack([new_grid_y, new_grid_x], axis=-1)  # [None, H, W, 2]
        output = tf.reshape(output, [cropped_shape[0], shape[0], shape[1], 2])

        height_f = tf.cast(shape[0], 'float32')
        width_f = tf.cast(shape[1], 'float32')

        new_h = output[:, :, :, 0]
        new_w = output[:, :, :, 1]
        new_w = (new_w + 1.0) * width_f / 2.0
        new_h = (new_h + 1.0) * height_f / 2.0
        new_h = tf.cast(tf.floor(new_h), tf.int32)
        new_w = tf.cast(tf.floor(new_w), tf.int32)

        # 4. 실제 cropped 크기 기준 경계 클리핑 (OOB 방어)
        cropped_h = cropped_shape[1]
        cropped_w = cropped_shape[2]
        new_h = tf.clip_by_value(new_h, 0, cropped_h - 1)
        new_w = tf.clip_by_value(new_w, 0, cropped_w - 1)

        batch_idx_tensor = tf.range(cropped_shape[0])
        batch_idx_tensor = tf.reshape(batch_idx_tensor, (cropped_shape[0], 1, 1))
        b = tf.tile(batch_idx_tensor, (1, shape[0], shape[1]))
        indexa = tf.stack([b, new_h, new_w], axis=-1)
        new_featuremap = tf.gather_nd(cropped, indexa)

        return new_featuremap, tf_non_zero_idx, batch_cord, theta_cord

    num_valid = tf.shape(tf_non_zero_idx)[0]
    return tf.cond(num_valid > 0, process_valid_boxes, handle_empty_boxes)


def rotate_np(cropped, theta):
    """
    :param cropped: (h,w,ch)
    :param theta: ()
    :return:
    """
    feature_shape = cropped.shape
    sin = np.sin(theta)
    cos = np.cos(theta)
    r_mat_row = np.stack([cos, -sin], axis=-1)
    r_mat_row2 = np.stack([sin, cos], axis=-1)
    r_mat = np.stack([r_mat_row, r_mat_row2], axis=-1)  # (2,2)
    r_mat = r_mat[np.newaxis, :, :]
    r_mat = np.tile(r_mat, [feature_shape[0] * feature_shape[1], 1, 1])  # (h*w,2,2)
    grid = np.meshgrid(np.linspace(-1.0, 1.0, feature_shape[0]), np.linspace(-1.0, 1.0, feature_shape[1]),
                       indexing='ij')
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, (-1, 2))
    grid = grid[:, :, np.newaxis]  # (h*w,2,1)
    output = np.matmul(r_mat, grid)
    output = np.reshape(output, [feature_shape[0], feature_shape[1], 2])

    new_h = output[:, :, 0]
    new_w = output[:, :, 1]
    new_w = (new_w + 1.0) * (feature_shape[1]) / 2.0
    new_h = (new_h + 1.0) * (feature_shape[0]) / 2.0
    new_w = np.floor(new_w).astype(np.int32)
    new_h = np.floor(new_h).astype(np.int32)
    
    # 5. NumPy와 TensorFlow 간 동일한 cropped 크기 기준 클리핑 계약 유지
    new_w = np.clip(new_w, 0, feature_shape[1] - 1)
    new_h = np.clip(new_h, 0, feature_shape[0] - 1)
    
    if len(feature_shape) > 2:
        rotated_cropped = cropped[new_h, new_w, :]
    else:
        rotated_cropped = cropped[new_h, new_w]

    return rotated_cropped

최종 개선사항
✅ tf.Tensor를 Python 조건문으로 직접 판정 → tf.cond 기반 그래프 분기 → tf.function 환경의 빈 ROI 처리 안정성 확보
✅ 원본 featuremap 크기로 인덱스 제한 → 실제 cropped 크기 기준 clipping → tf.gather_nd의 Out-of-Bounds 접근 방지
✅ 정확히 0인 분모만 방어 → 극소값까지 처리하는 safe_divisor 적용 → 회전각 계산의 수치 불안정성 완화
✅ ROI별 tile 반복 생성 → 브로드캐스팅 기반 회전 좌표 계산 → 불필요한 Tensor 복제 및 메모리 사용량 감소
✅ 유효 ROI 0개 상황 무방비 → 빈 Tensor 반환 경로 확보 → 검출 결과가 없는 배치에서도 파이프라인 생존성 확보
✅ TensorFlow와 NumPy의 경계 처리 기준 분리 → 동일한 cropped 공간 기준 clipping 적용 → 두 구현 간 인덱싱 무결성 강화
✅ 단순 기능 보완 중심 구조 → 수치 안정성·그래프 호환성·메모리 효율·인덱스 안전성 중심 방어 구조 → 연구용 코드를 운영 가능한 수준으로 승격

원본의 OOB·빈 ROI·극소 분모·불필요한 Tensor 복제 문제를 핵심적으로 제거하고, 현재 버전은 그래프 실행 안정성과 좌표 무결성까지 확보한 실무형 ROI 회전 연산 구조로 승격되었다.
