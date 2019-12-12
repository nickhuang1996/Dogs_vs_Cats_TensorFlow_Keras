from data_utils.get_file import get_file


def load_weights(net, model, include_top=True, cache_dir=None):

    if net['WEIGHTS'] == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    net['WEIGHTS_PATH'],
                                    cache_subdir='pretrained_models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb',
                                    cache_dir=cache_dir)
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    net['WEIGHTS_PATH_NO_TOP'],
                                    cache_subdir='pretrained_models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af',
                                    cache_dir=cache_dir)
        model.load_weights(weights_path)
    elif net['WEIGHTS'] is not None:
        model.load_weights(net['WEIGHTS'])

    return model
