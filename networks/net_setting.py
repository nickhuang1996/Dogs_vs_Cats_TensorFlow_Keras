from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras.applications.imagenet_utils import _obtain_input_shape
from tensorflow.python.keras._impl.keras.engine.network import get_source_inputs
from tensorflow.python.keras._impl.keras.layers import Input
from tensorflow.python.keras._impl.keras.models import Model

from parse_setting import args

from networks.pretrained_net.RESNET50.ResNet50 import ResNet50
from networks.pretrained_net.VGG16.VGG16 import VGG16
from networks.pretrained_net.INCEPTIONV3.InceptionV3 import InceptionV3
from networks.pretrained_net.VGG16.load_weights import load_weights as vgg16_load_weights
from networks.pretrained_net.RESNET50.load_weights import load_weights as resnet50_load_weights
from networks.pretrained_net.INCEPTIONV3.load_weights import load_weights as inceptionv3_load_weights

from networks.custom_net.VGG16BN import VGG16BN

NET_FACTORY = {
    'RESNET50': {
        'WEIGHTS_PATH': 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5',
        'WEIGHTS_PATH_NO_TOP': 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
        'DEFAULT_SIZE': 224,
        'MIN_SIZE': 197,
        'CLASS': ResNet50,
        'NAME': 'resnet50',
        'LOAD_WEIGHTS': resnet50_load_weights,
        'WEIGHTS': 'imagenet',
    },
    'VGG16': {
        'WEIGHTS_PATH': 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5',
        'WEIGHTS_PATH_NO_TOP': 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
        'DEFAULT_SIZE': 224,
        'MIN_SIZE': 48,
        'CLASS': VGG16,
        'NAME': 'vgg16',
        'LOAD_WEIGHTS': vgg16_load_weights,
        'WEIGHTS': 'imagenet',
    },
    'VGG16BN': {
        'DEFAULT_SIZE': 224,
        'MIN_SIZE': 48,
        'CLASS': VGG16BN,
        'NAME': 'vgg16_bn',
        'LOAD_WEIGHTS': None,
        'WEIGHTS': None,
    },
    'INCEPTIONV3': {
        'WEIGHTS_PATH': 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
        'WEIGHTS_PATH_NO_TOP': 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
        'DEFAULT_SIZE': 299,
        'MIN_SIZE': 139,
        'CLASS': InceptionV3,
        'NAME': 'inception_v3',
        'LOAD_WEIGHTS': inceptionv3_load_weights,
        'WEIGHTS': 'imagenet',
    },
}


def net_setting(include_top=True,
                cache_dir=args.pretrained_models_dir,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    if not (NET_FACTORY[args.use_network]['WEIGHTS'] in {'imagenet', None} or os.path.exists(NET_FACTORY[args.use_network]['WEIGHTS'])):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if NET_FACTORY[args.use_network]['WEIGHTS'] == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    if args.use_default_size is True:
        default_size = NET_FACTORY[args.use_network]['DEFAULT_SIZE']
    else:
        default_size = args.CenterCropSize[0]
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=NET_FACTORY[args.use_network]['MIN_SIZE'],
        data_format=K.image_data_format(),
        require_flatten=include_top,
        weights=NET_FACTORY[args.use_network]['WEIGHTS'])

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # load architecture
    net = NET_FACTORY[args.use_network]['CLASS'](
        include_top=include_top,
        classes=classes,
        pooling=pooling
    )
    x = net(img_input=img_input)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name=NET_FACTORY[args.use_network]['NAME'])
    # load weights
    if NET_FACTORY[args.use_network]['LOAD_WEIGHTS'] is not None:
        model = NET_FACTORY[args.use_network]['LOAD_WEIGHTS'](net=NET_FACTORY[args.use_network],
                                                              model=model,
                                                              include_top=include_top,
                                                              cache_dir=cache_dir)
    return model

