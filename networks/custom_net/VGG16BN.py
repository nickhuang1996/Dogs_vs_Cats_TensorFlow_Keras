from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras.layers import Activation

from tensorflow.python.keras import backend as K

block_settings = {
    1: [64, 2],
    2: [128, 2],
    3: [256, 3],
    4: [512, 3],
    5: [512, 3],
}


class VGG16BN(object):
    def __init__(self, include_top=True, classes=1000, pooling=None):

        self.include_top = include_top
        self.pooling = pooling

        self.block1_list = self._make_block(block_index=1)
        self.block2_list = self._make_block(block_index=2)
        self.block3_list = self._make_block(block_index=3)
        self.block4_list = self._make_block(block_index=4)
        self.block5_list = self._make_block(block_index=5)

        self.flatten = Flatten(name='flatten')
        self.layer14 = Dense(4096, activation='relu', name='fc1')
        self.layer15 = Dense(4096, activation='relu', name='fc2')
        self.layer16 = Dense(classes, activation='softmax', name='predictions')

        self.GAP = GlobalAveragePooling2D()
        self.GMP = GlobalMaxPooling2D()

    @staticmethod
    def _make_block(block_index):
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        layer_list = []
        for i in range(block_settings[block_index][1]):
            layer_list.append(Conv2D(block_settings[block_index][0],
                                     (3, 3),
                                     padding='same',
                                     name='block{}_conv{}'.format(block_index, i + 1)
                                     )
                              )
            layer_list.append(BatchNormalization(axis=bn_axis,
                                                 name='block{}_bn{}'.format(block_index, i + 1)
                                                 )
                              )
            layer_list.append(Activation('relu'))
        layer_list.append(MaxPooling2D((2, 2),
                                       strides=(2, 2),
                                       name='block{}_pool'.format(block_index)))
        return layer_list

    def block1(self, x):
        for i in range(len(self.block1_list)):
            x = self.block1_list[i](x)
        return x

    def block2(self, x):
        for i in range(len(self.block2_list)):
            x = self.block2_list[i](x)
        return x

    def block3(self, x):
        for i in range(len(self.block3_list)):
            x = self.block3_list[i](x)
        return x

    def block4(self, x):
        for i in range(len(self.block4_list)):
            x = self.block4_list[i](x)
        return x

    def block5(self, x):
        for i in range(len(self.block5_list)):
            x = self.block5_list[i](x)
        return x

    def __call__(self, img_input):
        x = self.block1(img_input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        if self.include_top:
            x = self.flatten(x)
            x = self.layer14(x)
            x = self.layer15(x)
            x = self.layer16(x)
        else:
            if self.pooling == 'avg':
                x = self.GAP(x)
            elif self.pooling == 'max':
                x = self.GMP(x)
        return x

