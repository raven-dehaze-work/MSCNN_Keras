"""
网络模型搭建
"""
import os
from keras import backend as K
import tensorflow as tf
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, BatchNormalization
from keras.layers.merge import concatenate


def build_net_model():
    """
    建立paper中的MSCNN
    :return: corseModel和fineModel的元祖
    """
    img_height = 240
    img_width = 320
    channel = 3

    # 通用输入
    input_img = Input((img_height, img_width, channel))
    coarseNet = _build_coarseNet(input_img)
    fineNet = _build_fineNet(input_img, coarseNet)

    # 建立coarse Model和fine Model
    coarseModel = Model(inputs=input_img, outputs=coarseNet)
    fineModel = Model(inputs=input_img, outputs=fineNet)

    # summary
    coarseModel.summary()
    fineModel.summary()

    return (coarseModel, fineModel)


def _build_coarseNet(input_img):
    """
    建立coarseNet
    :param input_img: 输入图片的tensor
    :return: coarseNet
    """
    conv1 = Conv2D(5, (11, 11), padding='same', activation='relu', name='coarseNet/conv1')(input_img)
    pool1 = MaxPooling2D((2, 2), name='coarseNet/pool1')(conv1)
    upsample1 = UpSampling2D((2, 2), name='coarseNet/upsample1')(pool1)
    normalize1 = BatchNormalization(axis=3, name='coarseNet/bn1')(upsample1)
    # dropout1 = Dropout(0.5, name='coarseNet/dropout1')(normalize1)

    conv2 = Conv2D(5, (9, 9), padding='same', activation='relu', name='coarseNet/conv2')(normalize1)
    pool2 = MaxPooling2D((2, 2), name='coarseNet/pool2')(conv2)
    upsample2 = UpSampling2D((2, 2), name='coarseNet/upsample2')(pool2)
    normalize2 = BatchNormalization(axis=3, name='coarseNet/bn2')(upsample2)
    # dropout2 = Dropout(0.5, name='coarseNet/dropout2')(normalize2)

    conv3 = Conv2D(10, (7, 7), padding='same', activation='relu', name='coarseNet/conv3')(normalize2)
    pool3 = MaxPooling2D((2, 2), name='coarseNet/pool3')(conv3)
    upsample3 = UpSampling2D((2, 2), name='coarseNet/upsample3')(pool3)
    # dropout3 = Dropout(0.5, name='coarseNet/dropout3')(upsample3)

    linear = LinearCombine(1, name='coarseNet/linear_combine')(upsample3)
    return linear


def _build_fineNet(input_img, coarseNet):
    """
    建立fineNet
    :param input_img: 输入图片的tensor
    :param coarseNet: coarseNet的Tensor
    :return: fineNet
    """
    # paper中的fine net 卷积kernel为4. 但经查看作者提供的源代码，第一层设置的6
    conv1 = Conv2D(6, (7, 7), padding='same', activation='relu', name='fineNet/conv1')(input_img)
    pool1 = MaxPooling2D((2, 2), name='fineNet/pool1')(conv1)
    upsample1 = UpSampling2D((2, 2), name='fineNet/upsample1')(pool1)

    # 级联coarseNet
    concat = concatenate([upsample1, coarseNet], axis=3, name='concat')
    normalize1 = BatchNormalization(axis=3, name='fineNet/bn1')(concat)
    # dropout1 = Dropout(0.5, name='fineNet/dropout1')(normalize1)

    conv2 = Conv2D(5, (5, 5), padding='same', activation='relu',name='fineNet/conv2')(normalize1)
    pool2 = MaxPooling2D((2, 2), name='fineNet/pool2')(conv2)
    upsample2 = UpSampling2D((2, 2), name='fineNet/upsample2')(pool2)
    normalize2 = BatchNormalization(axis=3, name='fineNet/bn2')(upsample2)
    # dropout2 = Dropout(0.5, name='fineNet/dropout2')(normalize2)

    conv3 = Conv2D(10, (3, 3), padding='same', activation='relu',name='fineNet/conv3')(normalize2)
    pool3 = MaxPooling2D((2, 2), name='fineNet/pool3')(conv3)
    upsample3 = UpSampling2D((2, 2), name='fineNet/upsample3')(pool3)
    # dropout3 = Dropout(0.5, name='fineNet/dropout3')(upsample3)

    linear = LinearCombine(1, name='fineNet/linear_combine')(upsample3)
    return linear


class LinearCombine(Layer):
    """
    paper 中的线性结合层
    """

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(LinearCombine, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      trainable=True,
                                      shape=(input_shape[3],),
                                      initializer='uniform')
        self.biases = self.add_weight(name='bias',
                                      trainable=True,
                                      shape=(self.output_dim,),
                                      initializer='normal')
        super(LinearCombine, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        out = K.bias_add(
            K.sum(tf.multiply(inputs, self.kernel), axis=3),
            self.biases
        )
        out = K.expand_dims(out, axis=3)
        return K.sigmoid(out)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.output_dim)
