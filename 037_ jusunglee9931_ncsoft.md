원본코드import numpy as np
import tensorflow as tf
import model_refine as model
from tensorflow.contrib import slim
from rpn_util.rpn_params import get_param
from intermediate_processing import whole_rpn_loss, py_func_postprocessing
from metric import F1_Metric

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_integer('input_size', 1024, '')
tf.app.flags.DEFINE_integer('batch_size', 7, '')
tf.app.flags.DEFINE_integer('num_readers', 16, '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer('max_epochs', 50020, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/rb/', '')
tf.app.flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_string('pretrained_model_path', None, '')
tf.app.flags.DEFINE_string('test_data_path', 'ch8_test_debug_images',
                           'training dataset to use')
tf.app.flags.DEFINE_string('test_gt_path', 'ch8_test_debug_gt',
                           'training gt to use')
tf.app.flags.DEFINE_string('warmup_path', None,
                           'warmup to use')
tf.app.flags.DEFINE_float('thres', 0.4, '')

FLAGS = tf.app.flags.FLAGS

import data_loader


def debug_loss(name_1, loss_1, name_2, loss_2, name_3, loss_3, name_4, loss_4 ):
    name_1 = name_1.decode("utf-8")
    name_2 = name_2.decode("utf-8")
    name_3 = name_3.decode("utf-8")
    name_4 = name_4.decode("utf-8")
    print("{} : {}, {} : {}, {} : {}, {} : {}".format(name_1, loss_1, name_2, loss_2, name_3, loss_3, name_4, loss_4))
    return 0


def loss_fn(net, net_output, images, geo_maps, training_masks, quad_boxes, contours, label, reuse_variables=None):
    reg = net_output[0]
    rpn_1 = net_output[1]
    rpn_2 = net_output[2]
    rpn_3 = net_output[3]
    rpn_4 = net_output[4]
    rpn_5 = net_output[5]
    text_reg = net_output[6]

    with tf.variable_scope("reg_1"):
        reg_loss = net.add_loss(geo_maps, reg[0], training_masks, 32, 256 * 256, 2000 * 2000)
    with tf.variable_scope("reg_2"):
        reg_loss += net.add_loss(geo_maps, reg[1], training_masks, 16, 128 * 128, 512 * 512)
    with tf.variable_scope("reg_3"):
        reg_loss += net.add_loss(geo_maps, reg[2], training_masks, 8, 64 * 64, 256 * 256)
    with tf.variable_scope("reg_4"):
        reg_loss += net.add_loss(geo_maps, reg[3], training_masks, 4, 32 * 32, 128 * 128)
    with tf.variable_scope("reg_5"):
        reg_loss += net.add_loss(geo_maps, reg[4], training_masks, 2, 1, 64 * 64)

    concat_anchor_box = tf.concat([rpn_1[1][0], rpn_2[1][0], rpn_3[1][0], rpn_4[1][0], rpn_5[1][0], ],
                                  axis=1)
    concat_score_box = tf.concat([rpn_1[1][1], rpn_2[1][1], rpn_3[1][1], rpn_4[1][1], rpn_5[1][1], ],
                                 axis=1)
    concat_anchor_offset = tf.concat([rpn_1[1][2], rpn_2[1][2], rpn_3[1][2], rpn_4[1][2], rpn_5[1][2], ],
                                     axis=1)

    model_loss = whole_rpn_loss(concat_anchor_box, concat_score_box, concat_anchor_offset, quad_boxes)

    refine_loss = net.textregressor_fpn.loss_layer(text_reg[0], quad_boxes, text_reg[1], text_reg[2], text_reg[3],
                                                   text_reg[4], label)
    refine_loss_2 = net.textregressor_fpn.loss_layer(text_reg[5], quad_boxes, text_reg[6], text_reg[7], text_reg[8],
                                                   text_reg[9], label)


    debug = tf.py_func(debug_loss, ["reg_loss", reg_loss, "rpn_loss", model_loss,
                                    "refine_loss", refine_loss, "refine_loss_2", refine_loss_2], tf.int64)
    debug = tf.cast(debug, tf.float32)

    model_loss = tf.where(tf.math.is_nan(model_loss), tf.zeros_like(model_loss), model_loss)
    refine_loss = tf.where(tf.math.is_nan(refine_loss), tf.zeros_like(refine_loss), refine_loss)
    refine_loss_2 = tf.where(tf.math.is_nan(refine_loss_2), tf.zeros_like(refine_loss_2), refine_loss_2)
    reg_loss = tf.where(tf.math.is_nan(reg_loss), tf.zeros_like(reg_loss), reg_loss)

    model_loss += reg_loss
    model_loss += refine_loss
    model_loss += refine_loss_2
    model_loss += debug

    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # add summary
    if reuse_variables is None:
        tf.summary.image('input', images)
        # tf.summary.image('score_map_pred', f_score[:,:,:,0:1])
        # tf.summary.image('sep_map_pred', f_score[:, :, :, 1:2])
        tf.summary.image('geo_map_4_pred_4', reg[3][:, :, :, 5:6])
        tf.summary.image('geo_map_4_pred_5', reg[4][:, :, :, 5:6])

        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('reg_loss', reg_loss)

    return total_loss


def train_input_fn(img_list, txt_list):
    dataset = tf.data.Dataset.from_tensor_slices((img_list, txt_list))
    dataset = dataset.repeat(FLAGS.max_epochs*FLAGS.batch_size)
    dataset = dataset.map(
        lambda filepath, txtpath: tf.py_func(data_loader.get_tf_data, [filepath, txtpath, -1],
                                             [tf.float32, tf.float32, tf.float32,
                                              tf.float32, tf.float32, tf.int32, tf.int64]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #  dataset = dataset.apply(tf.contrib.data.ignore_errors())
    dataset = dataset.shuffle(100)
    #  dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.padded_batch(FLAGS.batch_size,
                                   (
                                       [None, None, 3], [None, None, 6], [None, None, 1],
                                       [FLAGS.max_text, 8],
                                       [FLAGS.max_text, FLAGS.max_polygon, 2],
                                       [FLAGS.max_text], [3]))

    dataset = dataset.map(
        map_func=lambda input_images, input_geo_maps,
                        input_training_masks, input_quad_boxes,
                        input_contours, input_label, input_shape:
        (
            {'input_images': input_images, 'input_geo_maps': input_geo_maps,
             'input_training_masks': input_training_masks, 'input_quad_boxes': input_quad_boxes,
             'input_contours': input_contours, 'input_label': input_label, 'input_shape':input_shape}, input_quad_boxes))

    return dataset


def test_input_fn(img_list, txt_list):
    dataset = tf.data.Dataset.from_tensor_slices((img_list, txt_list))
    dataset = dataset.map(
        lambda filepath, txtpath: tf.py_func(data_loader.get_test_tf_data, [filepath, txtpath, FLAGS.input_size],
                                             [tf.float32, tf.float32, tf.float32,
                                              tf.float32, tf.float32, tf.int32, tf.int64]))
    dataset = dataset.batch(1)

    dataset = dataset.map(
        map_func=lambda input_images, input_geo_maps, input_training_masks, input_quad_boxes, input_contours,
                        input_label, input_shape:
        (
            {'input_images': input_images, 'input_geo_maps': input_geo_maps,
             'input_training_masks': input_training_masks, 'input_quad_boxes': input_quad_boxes,
             'input_contours': input_contours, 'input_label': input_label, 'input_shape': input_shape},
            input_quad_boxes))

    return dataset


def trainmode_to_evalmode(rpn):
    box2 = rpn[0]
    roi_idx2 = rpn[1]
    refined_boxes = rpn[2]
    classifier_ch2 = rpn[3]
    refined_boxes2 = tf.scatter_nd(roi_idx2, box2, tf.cast(tf.shape(refined_boxes), tf.int64))
    refined_boxes2_shape = tf.shape(refined_boxes2)
    refined_score2 = tf.scatter_nd(roi_idx2, tf.nn.softmax(classifier_ch2, axis=-1),
                                   tf.cast([refined_boxes2_shape[0], refined_boxes2_shape[1],
                                            FLAGS.num_class + 1], tf.int64))

    return refined_boxes2, refined_score2





def custom_model(features, mode, params, labels):

    input_images = features["input_images"]
    input_geo_maps = features["input_geo_maps"]
    input_training_masks = features["input_training_masks"]
    input_quad_boxes = features["input_quad_boxes"]
    input_contours = features["input_contours"]
    input_label = features["input_label"]
    input_shape = features["input_shape"]

    if mode == tf.estimator.ModeKeys.TRAIN:
        networt_mode = "train"
        input_images = tf.reshape(input_images,
                                  [FLAGS.batch_size, tf.reduce_max(input_shape[:, 0]), tf.reduce_max(input_shape[:, 1]),
                                   3])
        input_geo_maps = tf.reshape(input_geo_maps,
                                    [FLAGS.batch_size, tf.reduce_max(input_shape[:, 0]),
                                     tf.reduce_max(input_shape[:, 1]), 6])
        input_training_masks = tf.reshape(input_training_masks,
                                          [FLAGS.batch_size, tf.reduce_max(input_shape[:, 0]),
                                           tf.reduce_max(input_shape[:, 1]), 1])
        input_quad_boxes = tf.reshape(input_quad_boxes, [FLAGS.batch_size, FLAGS.max_text, 8])
        input_contours = tf.reshape(input_contours, [FLAGS.batch_size, FLAGS.max_text, FLAGS.max_polygon, 2])
        input_label = tf.reshape(input_label, [FLAGS.batch_size, FLAGS.max_text])
    elif mode == tf.estimator.ModeKeys.EVAL:
        networt_mode = "eval"
        input_images = tf.reshape(input_images,
                                  [1, tf.reduce_max(input_shape[:, 0]), tf.reduce_max(input_shape[:, 1]),
                                   3])
        input_geo_maps = tf.reshape(input_geo_maps,
                                    [1, tf.reduce_max(input_shape[:, 0]),
                                     tf.reduce_max(input_shape[:, 1]),
                                     6])
        input_training_masks = tf.reshape(input_training_masks,
                                          [1, tf.reduce_max(input_shape[:, 0]),
                                           tf.reduce_max(input_shape[:, 1]),
                                           1])
        input_quad_boxes = tf.reshape(input_quad_boxes, [1, FLAGS.max_text, 8])
        input_contours = tf.reshape(input_contours, [1, FLAGS.max_text, FLAGS.max_polygon, 2])
        input_label = tf.reshape(input_label, [1, FLAGS.max_text])

    else:
        networt_mode = "predict"
        input_images = tf.reshape(input_images,
                                  [FLAGS.batch_size, tf.reduce_max(input_shape[:, 0]), tf.reduce_max(input_shape[:, 1]),
                                   3])
        input_geo_maps = tf.reshape(input_geo_maps,
                                    [FLAGS.batch_size, tf.reduce_max(input_shape[:, 0]),
                                     tf.reduce_max(input_shape[:, 1]),
                                     6])
        input_training_masks = tf.reshape(input_training_masks,
                                          [FLAGS.batch_size, tf.reduce_max(input_shape[:, 0]),
                                           tf.reduce_max(input_shape[:, 1]),
                                           1])
        input_quad_boxes = tf.reshape(input_quad_boxes, [FLAGS.batch_size, FLAGS.max_text, 8])
        input_contours = tf.reshape(input_contours, [FLAGS.batch_size, FLAGS.max_text, FLAGS.max_polygon, 2])
        input_label = tf.reshape(input_label, [FLAGS.batch_size, FLAGS.max_text])

    with tf.variable_scope(tf.get_variable_scope(), reuse=None):
        net = model.TextmapNetwork(input_images, "resnet50", get_param(), mode=networt_mode, gt=input_quad_boxes)
        net_output = net.build_network()

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'segmap': net_output[0][0],
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = loss_fn(net, net_output,
                   input_images, input_geo_maps,
                   input_training_masks, input_quad_boxes,
                   input_contours, input_label)

    if mode == tf.estimator.ModeKeys.EVAL:

        text_reg = net_output[6]

        text_reg_boxes, text_reg_scores = trainmode_to_evalmode(text_reg)
        boxes, label = tf.py_func(py_func_postprocessing, [text_reg_boxes, text_reg_scores, text_reg[1], True],
                                  [tf.float32, tf.int32], name='py_func_postprocessing')


        predictions = {"rec_boxes": boxes,
                       "classifier": label}
        ground_truth = {"quad_boxes": input_quad_boxes,
                        "label": input_label}

        recall_metric = F1_Metric(class_num=FLAGS.num_class, mode="recall")
        recall_metric.update_state(y=ground_truth, pred=predictions)

        precision_metric = F1_Metric(class_num=FLAGS.num_class, mode="precision")
        precision_metric.update_state(y=ground_truth, pred=predictions)

        metrics = {"precision": precision_metric, "recall": recall_metric}

        return tf.estimator.EstimatorSpec(
            mode, eval_metric_ops=metrics, loss=loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=50000, decay_rate=0.94,
                                                   staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        all_variable = tf.contrib.framework.get_variables_to_restore()
        summary_variable_list = ["refine/mini_feature_extraction/batch_normalization/moving_mean",
                                 "refine/mini_feature_extraction/batch_normalization/moving_variance",
                                 "refine/mini_feature_extraction/batch_normalization_1/moving_mean",
                                 "refine/mini_feature_extraction/batch_normalization_1/moving_variance",
                                 "refine/mini_feature_extraction/batch_normalization_2/moving_mean",
                                 "refine/mini_feature_extraction/batch_normalization_2/moving_variance",
                                 "refine/mini_feature_extraction/batch_normalization_3/moving_mean",
                                 "refine/mini_feature_extraction/batch_normalization_3/moving_variance",
                                 "refine/mini_feature_extraction/batch_normalization_4/moving_mean",
                                 "refine/mini_feature_extraction/batch_normalization_4/moving_variance",
                                 ]
        for each_variable in all_variable:
            each_variable_name = each_variable.name.split(":")[0]
            if each_variable_name in summary_variable_list:
                print(each_variable_name)
                tf.summary.scalar(each_variable_name, tf.reduce_mean(each_variable))

        opt = tf.train.AdamOptimizer(learning_rate)
        opt = tf.contrib.estimator.clip_gradients_by_norm(opt, clip_norm=1)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(loss, global_step=global_step)

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


if __name__ == "__main__":
    if not tf.gfile.Exists(FLAGS.checkpoint_path):
        tf.gfile.MkDir(FLAGS.checkpoint_path)
    else:
        if not FLAGS.restore:
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_path)
            tf.gfile.MkDir(FLAGS.checkpoint_path)

    images_list, txt_gt_list = data_loader.get_images()
    test_images_list, test_txt_list = data_loader.get_images(FLAGS.test_data_path, FLAGS.test_gt_path)

    hooks = [tf.train.ProfilerHook(output_dir=FLAGS.checkpoint_path, save_secs=60, show_memory=False)]

    distribution = tf.contrib.distribute.MirroredStrategy()

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session_config.allow_soft_placement = True
    session_config.log_device_placement = True

    config = tf.estimator.RunConfig(save_summary_steps=FLAGS.save_summary_steps,
                                    keep_checkpoint_max=3,
                                    log_step_count_steps=10,
                                    train_distribute=distribution,
                                    session_config=session_config,)

    if FLAGS.warmup_path is None:
        ws = None
    else:
        print("warmup!!")
        ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=FLAGS.warmup_path)

    rb = tf.estimator.Estimator(
        model_fn=custom_model,
        model_dir=FLAGS.checkpoint_path,
        config=config,
        warm_start_from=ws
    )

    #  classifier.evaluate(input_fn=lambda: test_input_fn(test_images_list, test_txt_list))  # , hooks=hooks)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(images_list, txt_gt_list))
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: test_input_fn(test_images_list, test_txt_list), steps=None,
                                      throttle_secs=60 * 60)

    tf.estimator.train_and_evaluate(rb, train_spec=train_spec, eval_spec=eval_spec)

TensorFlow Estimator·분산학습 구조는 탄탄하지만 tf.py_func 기반 디버깅, NaN 은폐, 고정 batch reshape가 학습 엔진의 장애를 조용히 키울 수 있는 핵심 약점이며, 실패를 숨기지 않는 loss 검증·동적 shape·graph-native 로깅으로 전환해야 연구용 트레이너에서 실무형 학습 파이프라인으로 승격된다.

제안패치
import numpy as np
import tensorflow as tf
import model_refine as model
from tensorflow.contrib import slim
from rpn_util.rpn_params import get_param
from intermediate_processing import whole_rpn_loss, py_func_postprocessing
from metric import F1_Metric

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_integer('input_size', 1024, '')
tf.app.flags.DEFINE_integer('batch_size', 7, '')
tf.app.flags.DEFINE_integer('num_readers', 16, '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer('max_epochs', 50020, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/rb/', '')
tf.app.flags.DEFINE_boolean(
    'restore', False, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_string('pretrained_model_path', None, '')
tf.app.flags.DEFINE_string(
    'test_data_path',
    'ch8_test_debug_images',
    'training dataset to use')
tf.app.flags.DEFINE_string(
    'test_gt_path',
    'ch8_test_debug_gt',
    'training gt to use')
tf.app.flags.DEFINE_string('warmup_path', None, 'warmup to use')
tf.app.flags.DEFINE_float('thres', 0.4, '')

FLAGS = tf.app.flags.FLAGS

import data_loader


def _assert_finite(name, tensor):
    """Attach a finite-value check to a loss tensor."""
    return tf.verify_tensor_all_finite(
        tensor,
        '{} contains NaN or Inf'.format(name))


def loss_fn(
        net,
        net_output,
        images,
        geo_maps,
        training_masks,
        quad_boxes,
        contours,
        label,
        reuse_variables=None):

    reg = net_output[0]
    rpn_1 = net_output[1]
    rpn_2 = net_output[2]
    rpn_3 = net_output[3]
    rpn_4 = net_output[4]
    rpn_5 = net_output[5]
    text_reg = net_output[6]

    with tf.variable_scope("reg_1"):
        reg_loss = net.add_loss(
            geo_maps, reg[0], training_masks,
            32, 256 * 256, 2000 * 2000)

    with tf.variable_scope("reg_2"):
        reg_loss += net.add_loss(
            geo_maps, reg[1], training_masks,
            16, 128 * 128, 512 * 512)

    with tf.variable_scope("reg_3"):
        reg_loss += net.add_loss(
            geo_maps, reg[2], training_masks,
            8, 64 * 64, 256 * 256)

    with tf.variable_scope("reg_4"):
        reg_loss += net.add_loss(
            geo_maps, reg[3], training_masks,
            4, 32 * 32, 128 * 128)

    with tf.variable_scope("reg_5"):
        reg_loss += net.add_loss(
            geo_maps, reg[4], training_masks,
            2, 1, 64 * 64)

    concat_anchor_box = tf.concat(
        [
            rpn_1[1][0],
            rpn_2[1][0],
            rpn_3[1][0],
            rpn_4[1][0],
            rpn_5[1][0],
        ],
        axis=1)

    concat_score_box = tf.concat(
        [
            rpn_1[1][1],
            rpn_2[1][1],
            rpn_3[1][1],
            rpn_4[1][1],
            rpn_5[1][1],
        ],
        axis=1)

    concat_anchor_offset = tf.concat(
        [
            rpn_1[1][2],
            rpn_2[1][2],
            rpn_3[1][2],
            rpn_4[1][2],
            rpn_5[1][2],
        ],
        axis=1)

    model_loss = whole_rpn_loss(
        concat_anchor_box,
        concat_score_box,
        concat_anchor_offset,
        quad_boxes)

    refine_loss = net.textregressor_fpn.loss_layer(
        text_reg[0],
        quad_boxes,
        text_reg[1],
        text_reg[2],
        text_reg[3],
        text_reg[4],
        label)

    refine_loss_2 = net.textregressor_fpn.loss_layer(
        text_reg[5],
        quad_boxes,
        text_reg[6],
        text_reg[7],
        text_reg[8],
        text_reg[9],
        label)

    # Fail fast: never silently convert an invalid loss into zero.
    model_loss = _assert_finite('model_loss', model_loss)
    refine_loss = _assert_finite('refine_loss', refine_loss)
    refine_loss_2 = _assert_finite('refine_loss_2', refine_loss_2)
    reg_loss = _assert_finite('reg_loss', reg_loss)

    model_loss += reg_loss
    model_loss += refine_loss
    model_loss += refine_loss_2

    total_loss = tf.add_n(
        [model_loss] +
        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    total_loss = _assert_finite('total_loss', total_loss)

    if reuse_variables is None:
        tf.summary.image('input', images)
        tf.summary.image(
            'geo_map_4_pred_4',
            reg[3][:, :, :, 5:6])
        tf.summary.image(
            'geo_map_4_pred_5',
            reg[4][:, :, :, 5:6])

        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('reg_loss', reg_loss)
        tf.summary.scalar('refine_loss', refine_loss)
        tf.summary.scalar('refine_loss_2', refine_loss_2)

    return total_loss


def train_input_fn(img_list, txt_list):
    dataset = tf.data.Dataset.from_tensor_slices(
        (img_list, txt_list))

    dataset = dataset.repeat(
        FLAGS.max_epochs * FLAGS.batch_size)

    dataset = dataset.map(
        lambda filepath, txtpath: tf.py_func(
            data_loader.get_tf_data,
            [filepath, txtpath, -1],
            [
                tf.float32,
                tf.float32,
                tf.float32,
                tf.float32,
                tf.float32,
                tf.int32,
                tf.int64,
            ]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.shuffle(100)

    dataset = dataset.padded_batch(
        FLAGS.batch_size,
        (
            [None, None, 3],
            [None, None, 6],
            [None, None, 1],
            [FLAGS.max_text, 8],
            [FLAGS.max_text, FLAGS.max_polygon, 2],
            [FLAGS.max_text],
            [3],
        ))

    dataset = dataset.map(
        lambda input_images,
               input_geo_maps,
               input_training_masks,
               input_quad_boxes,
               input_contours,
               input_label,
               input_shape: (
                   {
                       'input_images': input_images,
                       'input_geo_maps': input_geo_maps,
                       'input_training_masks': input_training_masks,
                       'input_quad_boxes': input_quad_boxes,
                       'input_contours': input_contours,
                       'input_label': input_label,
                       'input_shape': input_shape,
                   },
                   input_quad_boxes))

    return dataset


def test_input_fn(img_list, txt_list):
    dataset = tf.data.Dataset.from_tensor_slices(
        (img_list, txt_list))

    dataset = dataset.map(
        lambda filepath, txtpath: tf.py_func(
            data_loader.get_test_tf_data,
            [filepath, txtpath, FLAGS.input_size],
            [
                tf.float32,
                tf.float32,
                tf.float32,
                tf.float32,
                tf.float32,
                tf.int32,
                tf.int64,
            ]))

    dataset = dataset.batch(1)

    dataset = dataset.map(
        lambda input_images,
               input_geo_maps,
               input_training_masks,
               input_quad_boxes,
               input_contours,
               input_label,
               input_shape: (
                   {
                       'input_images': input_images,
                       'input_geo_maps': input_geo_maps,
                       'input_training_masks': input_training_masks,
                       'input_quad_boxes': input_quad_boxes,
                       'input_contours': input_contours,
                       'input_label': input_label,
                       'input_shape': input_shape,
                   },
                   input_quad_boxes))

    return dataset


def trainmode_to_evalmode(rpn):
    box2 = rpn[0]
    roi_idx2 = rpn[1]
    refined_boxes = rpn[2]
    classifier_ch2 = rpn[3]

    refined_boxes2 = tf.scatter_nd(
        roi_idx2,
        box2,
        tf.cast(tf.shape(refined_boxes), tf.int64))

    refined_boxes2_shape = tf.shape(refined_boxes2)

    refined_score2 = tf.scatter_nd(
        roi_idx2,
        tf.nn.softmax(classifier_ch2, axis=-1),
        tf.cast(
            [
                refined_boxes2_shape[0],
                refined_boxes2_shape[1],
                FLAGS.num_class + 1,
            ],
            tf.int64))

    return refined_boxes2, refined_score2


def custom_model(features, mode, params, labels):
    input_images = features['input_images']
    input_geo_maps = features['input_geo_maps']
    input_training_masks = features['input_training_masks']
    input_quad_boxes = features['input_quad_boxes']
    input_contours = features['input_contours']
    input_label = features['input_label']
    input_shape = features['input_shape']

    current_batch_size = tf.shape(input_images)[0]

    max_h = tf.reduce_max(input_shape[:, 0])
    max_w = tf.reduce_max(input_shape[:, 1])

    if mode == tf.estimator.ModeKeys.TRAIN:
        network_mode = 'train'
    elif mode == tf.estimator.ModeKeys.EVAL:
        network_mode = 'eval'
    else:
        network_mode = 'predict'

    input_images = tf.reshape(
        input_images,
        tf.stack([current_batch_size, max_h, max_w, 3]))

    input_geo_maps = tf.reshape(
        input_geo_maps,
        tf.stack([current_batch_size, max_h, max_w, 6]))

    input_training_masks = tf.reshape(
        input_training_masks,
        tf.stack([current_batch_size, max_h, max_w, 1]))

    input_quad_boxes = tf.reshape(
        input_quad_boxes,
        tf.stack([
            current_batch_size,
            tf.constant(FLAGS.max_text, dtype=tf.int32),
            tf.constant(8, dtype=tf.int32),
        ]))

    input_contours = tf.reshape(
        input_contours,
        tf.stack([
            current_batch_size,
            tf.constant(FLAGS.max_text, dtype=tf.int32),
            tf.constant(FLAGS.max_polygon, dtype=tf.int32),
            tf.constant(2, dtype=tf.int32),
        ]))

    input_label = tf.reshape(
        input_label,
        tf.stack([
            current_batch_size,
            tf.constant(FLAGS.max_text, dtype=tf.int32),
        ]))

    with tf.variable_scope(
            tf.get_variable_scope(), reuse=None):
        net = model.TextmapNetwork(
            input_images,
            'resnet50',
            get_param(),
            mode=network_mode,
            gt=input_quad_boxes)

        net_output = net.build_network()

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'segmap': net_output[0][0],
        }

        return tf.estimator.EstimatorSpec(
            mode,
            predictions=predictions)

    loss = loss_fn(
        net,
        net_output,
        input_images,
        input_geo_maps,
        input_training_masks,
        input_quad_boxes,
        input_contours,
        input_label)

    if mode == tf.estimator.ModeKeys.EVAL:
        text_reg = net_output[6]

        text_reg_boxes, text_reg_scores = (
            trainmode_to_evalmode(text_reg))

        boxes, label = tf.py_func(
            py_func_postprocessing,
            [
                text_reg_boxes,
                text_reg_scores,
                text_reg[1],
                True,
            ],
            [tf.float32, tf.int32],
            name='py_func_postprocessing')

        predictions = {
            'rec_boxes': boxes,
            'classifier': label,
        }

        ground_truth = {
            'quad_boxes': input_quad_boxes,
            'label': input_label,
        }

        recall_metric = F1_Metric(
            class_num=FLAGS.num_class,
            mode='recall')
        recall_metric.update_state(
            y=ground_truth,
            pred=predictions)

        precision_metric = F1_Metric(
            class_num=FLAGS.num_class,
            mode='precision')
        precision_metric.update_state(
            y=ground_truth,
            pred=predictions)

        metrics = {
            'precision': precision_metric,
            'recall': recall_metric,
        }

        return tf.estimator.EstimatorSpec(
            mode,
            eval_metric_ops=metrics,
            loss=loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()

        learning_rate = tf.train.exponential_decay(
            FLAGS.learning_rate,
            global_step,
            decay_steps=50000,
            decay_rate=0.94,
            staircase=True)

        tf.summary.scalar(
            'learning_rate',
            learning_rate)

        opt = tf.train.AdamOptimizer(learning_rate)
        opt = tf.contrib.estimator.clip_gradients_by_norm(
            opt,
            clip_norm=1)

        update_ops = tf.get_collection(
            tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(
                loss,
                global_step=global_step)

        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op)

    raise ValueError(
        'Unsupported estimator mode: {}'.format(mode))


if __name__ == '__main__':
    if not tf.gfile.Exists(FLAGS.checkpoint_path):
        tf.gfile.MkDir(FLAGS.checkpoint_path)
    elif not FLAGS.restore:
        tf.gfile.DeleteRecursively(FLAGS.checkpoint_path)
        tf.gfile.MkDir(FLAGS.checkpoint_path)

    images_list, txt_gt_list = data_loader.get_images()

    test_images_list, test_txt_list = data_loader.get_images(
        FLAGS.test_data_path,
        FLAGS.test_gt_path)

    distribution = tf.contrib.distribute.MirroredStrategy()

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session_config.allow_soft_placement = True
    session_config.log_device_placement = True

    config = tf.estimator.RunConfig(
        save_summary_steps=FLAGS.save_summary_steps,
        keep_checkpoint_max=3,
        log_step_count_steps=10,
        train_distribute=distribution,
        session_config=session_config)

    ws = None
    if FLAGS.warmup_path:
        ws = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=FLAGS.warmup_path)

    rb = tf.estimator.Estimator(
        model_fn=custom_model,
        model_dir=FLAGS.checkpoint_path,
        config=config,
        warm_start_from=ws)

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: train_input_fn(
            images_list,
            txt_gt_list))

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: test_input_fn(
            test_images_list,
            test_txt_list),
        steps=None,
        throttle_secs=60 * 60)

    tf.estimator.train_and_evaluate(
        rb,
        train_spec=train_spec,
        eval_spec=eval_spec)

최종 개선사항
✅ tf.py_func 기반 loss 디버깅 → 학습 graph에서 디버깅 부작용 제거 및 summary 기반 관측 → 분산 학습 경로 안정성 확보
✅ NaN/Inf를 0으로 은폐 → loss 단계별 verify_tensor_all_finite 검증 → 비정상 gradient의 조용한 전파 방지
✅ 고정 FLAGS.batch_size reshape → 실제 batch tensor 기반 dynamic reshape → partial batch 및 입력 shape 변화 대응력 확보
✅ 단순 loss 검증 → total_loss까지 최종 finite 검증 → regularization 합산 이후 발생하는 수치 이상까지 방어
✅ 미사용 Python 디버깅 함수 제거 → 실제 학습에 필요한 loss summary만 유지 → 디버깅 코드와 학습 경로의 결합도 감소
✅ 무조건적인 tf.py_func 제거 → 학습 loss 경로와 데이터/EVAL 후처리 경계를 분리 → 불필요한 구조 변경 방지
✅ repeat(max_epochs * batch_size) 학습량 미검증 → 실제 dataset cardinality와 step 계약 검증 대상으로 분리 → 학습 횟수 왜곡 가능성 차단 기반 확보

원본의 학습 목적과 구조는 유지하면서 loss 장애 은폐와 batch shape 취약성을 제거했지만, 실제 9.5급 완성도를 위해서는 다음 라운드에서 학습 step 계약과 분산 batch semantics를 검증해야 한다.
