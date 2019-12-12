from tensorflow.python.keras.callbacks import TensorBoard


def get_log(args):
    tb_writer = TensorBoard(log_dir=args.experiment_dir + '/tensorboard')
    return tb_writer
