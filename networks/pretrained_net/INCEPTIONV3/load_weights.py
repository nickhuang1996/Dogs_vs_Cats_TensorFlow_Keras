from data_utils.get_file import get_file


def load_weights(net, model, include_top=True, cache_dir=None):
    if net['WEIGHTS'] == 'imagenet':
        if include_top:
            weights_path = get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
                net['WEIGHTS_PATH'],
                cache_subdir='pretrained_models',
                file_hash='9a0d58056eeedaa3f26cb7ebd46da564',
                cache_dir=cache_dir)
        else:
            weights_path = get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                net['WEIGHTS_PATH_NO_TOP'],
                cache_subdir='pretrained_models',
                file_hash='bcbd6486424b2319ff4ef7d526e38f63',
                cache_dir=cache_dir)
        model.load_weights(weights_path)
    elif net['WEIGHTS'] is not None:
        model.load_weights(net['WEIGHTS'])

    return model
