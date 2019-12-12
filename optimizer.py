from tensorflow.python.keras import optimizers


def get_optimizer(args):
    if args.optimizer == 'Adam':
        kwargs = dict(
            lr=args.lr,
            beta_1=args.beta1,
            beta_2=args.beta2,
            epsilon=args.epsilon
        )
        optimizer = optimizers.Adam(**kwargs)
    elif args.optimizer == 'SGD':
        kwargs = dict(
            lr=args.lr,
            momentum=args.momentum
        )
        optimizer = optimizers.SGD(**kwargs)
    else:
        raise NotImplementedError('Unsupported Optimizer {}'.format(args.optimizer))
    return optimizer
