from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from tensorflow.keras import Model
import logging

logger = logging.getLogger(__name__)


def resblock(x, num_filters, activation='elu', downsample=True, name=None, batch_norm=False):
    """

    :param x:
    :param num_filters:
    :param activation:
    :param downsample:
    :param name:
    :param batch_norm:
    :return:
    """

    if downsample:
        stride_input = 2
        shortcut = layers.Conv2D(num_filters, kernel_size=1,
                                 strides=stride_input,
                                 kernel_regularizer=l2(1e-4),
                                 name=name + '_shortcut',
                                 kernel_initializer='he_normal')(x)
    else:
        stride_input = 1
        shortcut = x
    x = layers.Activation(activation, name=name + '_actv1')(x)
    x = layers.Conv2D(num_filters, kernel_size=3,
                      strides=stride_input,
                      padding='same',
                      kernel_regularizer=l2(1e-4),
                      kernel_initializer='he_normal',
                      name=name + '_conv1', )(x)
    x = layers.Activation(activation, name=name + '_actv2')(x)
    x = layers.Conv2D(num_filters, kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_regularizer=l2(1e-4),
                      kernel_initializer='he_normal',
                      name=name + '_conv2', )(x)
    if batch_norm:
        x = layers.BatchNormalization(name=name + '_bn')(x)
    return layers.Add(name=name + '_add')([x, shortcut])


def stack(x, num_filters, num_resblocks, activation='elu', name=None, downsample=True, do_bn=False):
    """ The basic x -> actv -> conv -> actv -> conv --> add -->  block
                  |                                      ^
                  |______________________________________|
    """
    # first res block down samples
    x = resblock(x, num_filters=num_filters,
                 activation=activation,
                 downsample=downsample,
                 name=name + '_res0',
                 batch_norm=do_bn)
    for i in range(1, num_resblocks):
        x = resblock(x, num_filters=num_filters,
                     activation=activation, downsample=False,
                     name=name + f'_res{i}', batch_norm=do_bn)
    return x


def generate_model(input_shape=(32, 32, 3), activation='elu', add_batch_norm=False, depth=20, num_classes=10,
                   num_filters_layer0=16):
    """

    """
    resblocks_per_stack = (depth - 2) // 6
    num_filters = num_filters_layer0
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(num_filters, kernel_size=3, padding='same',
                      kernel_regularizer=l2(1e-4), kernel_initializer='he_normal', name='input_conv')(inputs)
    x = stack(x, num_filters, resblocks_per_stack, activation=activation,
              do_bn=add_batch_norm, downsample=False, name="stack0")
    for stack_i in (1, 2):
        num_filters *= 2
        x = stack(x, num_filters, resblocks_per_stack, activation=activation, do_bn=add_batch_norm,
                  name=f"stack{stack_i}")

    x = layers.Activation(activation, name='global_elu')(x)
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='fc')(x)
    return Model(inputs=inputs, outputs=outputs)
