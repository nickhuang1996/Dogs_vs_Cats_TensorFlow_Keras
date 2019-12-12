from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import EarlyStopping


def save_checkpoint(args):
    checkpoint = ModelCheckpoint(
        filepath=args.model_dir + '/best.h5',
        monitor='val_acc',
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
        period=args.epochs_per_val,
    )
    return checkpoint


def get_early_stopping(args=None):
    early_stopping = EarlyStopping(monitor='acc')
    return early_stopping
