from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras import backend as K
import numpy as np


def get_lr_scheduler(args):
    lr_scheduler = MultiStepLR(args=args)
    return lr_scheduler


class MultiStepLR(Callback):
    """Learning rate scheduler.

    Arguments:
        args: parser_setting
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, args, verbose=0):
        super(MultiStepLR, self).__init__()
        self.args = args
        self.steps = args.lr_decay_epochs
        self.factor = args.lr_decay_factor
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.schedule(epoch)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)
        print("learning rate: {:.7f}".format(K.get_value(self.model.optimizer.lr)).rstrip('0'))
        if self.verbose > 0:
            print('\nEpoch %05d: MultiStepLR reducing learning '
                  'rate to %s.' % (epoch + 1, lr))

    def schedule(self, epoch):
        lr = K.get_value(self.model.optimizer.lr)
        for i in range(len(self.steps)):
            if epoch == self.steps[i]:
                lr = lr * self.factor

        return lr
