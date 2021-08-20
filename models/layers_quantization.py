import tensorflow as tf
import utils


class Conv2D_quant(tf.keras.layers.Conv2D):
    def __init__(self, padding='SAME', quant_flage=False, **kwargs):
        super(Conv2D_quant, self).__init__(**kwargs)
        self.padding = padding
        self.quant_flage = quant_flage

    def call(self, inputs):
        if self.quant_flage:
            inputs, inputs_scale, inputs_zero_point = utils.quantize_int8_input(inputs)
            _, kernel, s_kernel, _ = utils.quantize_int8_conv2d(self.kernel)
            output = tf.nn.conv2d(input=(inputs-inputs_zero_point), filters=kernel, strides=self.strides, padding=self.padding)
            return output * inputs_scale * s_kernel
        else:
            output = tf.nn.conv2d(input=inputs, filters=self.kernel, strides=self.strides, padding=self.padding)
            return output

    def reset_quant_flage(self, quant_flage):
        self.quant_flage = quant_flage


class Dense_quant(tf.keras.layers.Dense):
    def __init__(self, quant_flage=False, **kwargs):
        super(Dense_quant, self).__init__(**kwargs)
        self.quant_flage = quant_flage

    def call(self, inputs):
        if self.quant_flage:
            inputs, inputs_scale, inputs_zero_point = utils.quantize_int8_input(inputs)
            _, kernel, s_kernel, _ = utils.quantize_int8_dense(self.kernel)
            output = tf.matmul((inputs-inputs_zero_point), kernel)
            return output * inputs_scale * s_kernel
        else:
            output = tf.matmul(inputs, self.kernel)
            return output

    def reset_quant_flage(self, quant_flage):
        self.quant_flage = quant_flage
