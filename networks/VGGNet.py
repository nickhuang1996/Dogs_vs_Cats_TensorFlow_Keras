from tensorflow.python.keras.layers import Flatten, Dense, Dropout
from networks.net_setting import net_setting


class VGGNet(object):
    def __init__(self, args):
        self.args = args
        self.features = net_setting(include_top=False,
                                    input_tensor=None,
                                    input_shape=(args.CenterCropSize[0],
                                                 args.CenterCropSize[1],
                                                 3))
        self.flatten = Flatten()
        self.dropout = Dropout(0.5)
        self.dense = Dense(2, activation='softmax', name='softmax')

    def __call__(self):
        x = self.features.output
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense(x)
        return x
