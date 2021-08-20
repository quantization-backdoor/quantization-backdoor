import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Activation
from models.layers_quantization import Conv2D_quant, Dense_quant
from tensorflow.keras import Model


class ResnetBlock(Model):

    def __init__(self, filters, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D_quant(filters=filters, kernel_size=(3, 3), strides=strides, padding='SAME', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D_quant(filters=filters, kernel_size=(3, 3), strides=1, padding='SAME', use_bias=False)
        self.b2 = BatchNormalization()

        if residual_path:
            self.down_c1 = Conv2D_quant(filters=filters, kernel_size=(1, 1), strides=strides, padding='SAME', use_bias=False)
            self.down_b1 = BatchNormalization()

        self.a2 = Activation('relu')

    def call(self, inputs):
        residual = inputs
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        out = self.a2(y + residual)
        return out

    def reset_quant_flage(self, quant_flage):
        self.c1.reset_quant_flage(quant_flage)
        self.c2.reset_quant_flage(quant_flage)
        if self.residual_path:
            self.down_c1.reset_quant_flage(quant_flage)


class Bottleneck(Model):

    def __init__(self, filters, strides=1, residual_path=False):
        super(Bottleneck, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D_quant(filters=filters, kernel_size=(1, 1), strides=1, padding='SAME', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D_quant(filters=filters, kernel_size=(3, 3), strides=strides, padding='SAME', use_bias=False)
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')

        self.c3 = Conv2D_quant(filters=filters * 4, kernel_size=(1, 1), strides=1, padding='SAME', use_bias=False)
        self.b3 = BatchNormalization()

        if residual_path:
            self.down_c1 = Conv2D_quant(filters=filters * 4, kernel_size=(1, 1), strides=strides, padding='SAME', use_bias=False)
            self.down_b1 = BatchNormalization()

        self.a3 = Activation('relu')

    def call(self, inputs):
        residual = inputs
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)

        x = self.c3(x)
        y = self.b3(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        out = self.a3(y + residual)
        return out

    def reset_quant_flage(self, quant_flage):
        self.c1.reset_quant_flage(quant_flage)
        self.c2.reset_quant_flage(quant_flage)
        self.c3.reset_quant_flage(quant_flage)
        if self.residual_path:
            self.down_c1.reset_quant_flage(quant_flage)


class ResNet(Model):

    def __init__(self, block_list, block_type, initial_filters=64, classes=10, quant_flage=False, type=0):
        super(ResNet, self).__init__()
        self.num_blocks = len(block_list)
        self.block_list = block_list
        self.out_filters = initial_filters
        self.classes = classes
        self.quant_flage = quant_flage

        self.c1 = Conv2D_quant(filters=self.out_filters, kernel_size=(3, 3), strides=1, padding='SAME', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.blocks = tf.keras.models.Sequential()
        for block_id in range(len(block_list)):
            for layer_id in range(block_list[block_id]):

                if block_id != 0 and layer_id == 0:
                    block = block_type(self.out_filters, strides=2, residual_path=True)
                elif block_id == 0 and layer_id == 0 and type:
                    block = block_type(self.out_filters, residual_path=True)
                else:
                    block = block_type(self.out_filters, residual_path=False)
                self.blocks.add(block)

            self.out_filters *= 2

        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = Dense_quant(units=self.classes, use_bias=False,)
        self.softmax = Activation('softmax')

    def call(self, inputs):
        inputs = tf.minimum(inputs, 1.)
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)
        x = self.f1(x)
        y = self.softmax(x)
        return y

    def reset_quant_flage(self, quant_flage):
        self.quant_flage = quant_flage

        self.c1.reset_quant_flage(quant_flage)
        for layer in self.blocks.layers:
            layer.reset_quant_flage(quant_flage)
        self.f1.reset_quant_flage(quant_flage)


def resnet18(classes=10):
    model = ResNet([2, 2, 2, 2], ResnetBlock, classes=classes, type=0)
    return model


def resnet50(classes=10):
    model = ResNet([3, 4, 6, 3], Bottleneck, classes=classes, type=1)
    return model


if __name__ == '__main__':
    resnet18()

