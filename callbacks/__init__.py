from callbacks.checkpoint import get_early_stopping
from callbacks.checkpoint import save_checkpoint
from callbacks.log import get_log
from callbacks.lr_scheduler import get_lr_scheduler


def get_callbacks(args):
    callbacks = []
    checkpoint = save_checkpoint(args=args)
    lr_scheduler = get_lr_scheduler(args=args)
    callbacks.append(lr_scheduler)
    callbacks.append(checkpoint)
    if args.use_early_stopping is True:
        early_stopping = get_early_stopping(args=args)
        callbacks.append(early_stopping)
    if args.use_tensorboard is True:
        tb_writer = get_log(args=args)
        callbacks.append(tb_writer)
    return callbacks
