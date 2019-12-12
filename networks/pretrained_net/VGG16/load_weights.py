from data_utils.get_file import get_file
from tensorflow.python.keras._impl.keras.utils import layer_utils
from tensorflow.python.keras._impl.keras import backend as K


def load_weights(net, model, include_top=True, cache_dir=None):

    if net['WEIGHTS'] == 'imagenet':
        if include_top:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                    net['WEIGHTS_PATH'],
                                    cache_subdir='pretrained_models',
                                    file_hash='64373286793e3c8b2b4e3219cbf3544b',
                                    cache_dir=cache_dir)
        else:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    net['WEIGHTS_PATH_NO_TOP'],
                                    cache_subdir='pretrained_models',
                                    file_hash='6d6bbae143d832006294945121d1f1fc',
                                    cache_dir=cache_dir)
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense,
                                                              shape,
                                                              'channels_first')

    elif net['WEIGHTS'] is not None:
        model.load_weights(net['WEIGHTS'])

    return model
