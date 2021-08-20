import tensorflow as tf


def quantize_int8_input(tensor):
    vmin = tf.reduce_min(tensor)
    vmax = tf.reduce_max(tensor)
    scale = (vmax - vmin) / 255.
    zero_point_quantize = -(vmin / scale + 128.)
    tensor_quantize = tensor / scale + zero_point_quantize
    tensor_round = round_through(tensor_quantize)
    zero_point_round = round_through(zero_point_quantize)
    return tensor_round, scale, zero_point_round


def quantize_int8_conv2d(tensor):
    abs_value = tf.abs(tensor)
    vmax = tf.reduce_max(abs_value, axis=[0, 1, 2])
    scale = tf.divide(vmax, 127.)
    tensor_quantize = tf.divide(tensor, scale)
    tensor_round = round_through(tensor_quantize)
    decimal = tensor_quantize - tf.round(tensor_quantize)
    return tensor_quantize, tensor_round, scale, decimal


def quantize_int8_dense(tensor):
    abs_value = tf.abs(tensor)
    vmax = tf.reduce_max(abs_value)
    scale = tf.divide(vmax, 127.)
    scale = tf.expand_dims(scale, axis=0)
    tensor_quantize = tf.divide(tensor, scale)
    tensor_round = round_through(tensor_quantize)
    decimal = tensor_quantize - tf.round(tensor_quantize)
    return tensor_quantize, tensor_round, scale, decimal


def rint_through(x):
    rinted = tf.compat.v1.rint(x)
    return x + tf.stop_gradient(rinted-x)


def round_through(x):
    rounded = tf.round(x)
    return x + tf.stop_gradient(rounded-x)


def keep_scale_dense(param, base_param):
    abs_base_param = tf.abs(base_param)
    max_base_param = tf.reduce_max(abs_base_param)
    new_param = tf.clip_by_value(tf.where(max_base_param - abs_base_param > 0, param, base_param),
                                 -max_base_param, max_base_param)
    return new_param


def keep_scale_conv2d(param, base_param):
    abs_base_param = tf.abs(base_param)
    max_base_param = tf.reduce_max(abs_base_param, axis=[0, 1, 2])
    new_param = tf.clip_by_value(tf.where(max_base_param - abs_base_param > 0, param, base_param),
                                 -max_base_param, max_base_param)
    return new_param

