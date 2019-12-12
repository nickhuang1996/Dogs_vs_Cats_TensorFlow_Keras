from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import AveragePooling2D
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras.layers import Dense


class BasicConv2d(object):
    def __init__(self,
                 filters,
                 num_row,
                 num_col,
                 padding='same',
                 strides=(1, 1),
                 name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 3

        self.conv = Conv2D(filters,
                           (num_row, num_col),
                           strides=strides,
                           padding=padding,
                           use_bias=False,
                           name=conv_name)
        self.bn = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)

        self.relu = Activation('relu', name=name)

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionA(object):
    def __init__(self, in_channels, pool_features, name, branch_pool_filters):
        self.in_channels = in_channels
        self.pool_features = pool_features
        self.name = name
        self.branch_pool_filters = branch_pool_filters

        if K.image_data_format() == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = 3

        self.branch1x1 = BasicConv2d(64, 1, 1)

        self.branch5x5_1 = BasicConv2d(48, 1, 1)
        self.branch5x5_2 = BasicConv2d(64, 5, 5)

        self.branch3x3dbl_1 = BasicConv2d(64, 1, 1)
        self.branch3x3dbl_2 = BasicConv2d(96, 3, 3)
        self.branch3x3dbl_3 = BasicConv2d(96, 3, 3)

        self.avgpool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')
        self.branch_pool = BasicConv2d(self.branch_pool_filters, 1, 1)

    def __call__(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.avgpool(x)
        branch_pool = self.branch_pool(branch_pool)

        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=self.channel_axis,
            name=self.name)

        return x


class InceptionB(object):
    def __init__(self, in_channels, name):
        self.in_channels = in_channels
        self.name = name

        if K.image_data_format() == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = 3

        self.branch3x3 = BasicConv2d(384, 3, 3, strides=(2, 2), padding='valid')

        self.branch3x3dbl_1 = BasicConv2d(64, 1, 1)
        self.branch3x3dbl_2 = BasicConv2d(96, 3, 3)
        self.branch3x3dbl_3 = BasicConv2d(96, 3, 3, strides=(2, 2), padding='valid')

        self.maxpool = MaxPooling2D((3, 3), strides=(2, 2))

    def __call__(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.maxpool(x)
        x = layers.concatenate(
            [branch3x3, branch3x3dbl, branch_pool], axis=self.channel_axis, name=self.name)

        return x


class InceptionC(object):
    def __init__(self, in_channels, channels_7x7, name):
        self.in_channels = in_channels
        self.channels_7x7 = channels_7x7  # 128
        self.name = name

        if K.image_data_format() == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = 3

        self.branch1x1 = BasicConv2d(192, 1, 1)

        self.branch7x7_1 = BasicConv2d(self.channels_7x7, 1, 1)
        self.branch7x7_2 = BasicConv2d(self.channels_7x7, 1, 7)
        self.branch7x7_3 = BasicConv2d(192, 7, 1)

        self.branch7x7dbl_1 = BasicConv2d(self.channels_7x7, 1, 1)
        self.branch7x7dbl_2 = BasicConv2d(self.channels_7x7, 7, 1)
        self.branch7x7dbl_3 = BasicConv2d(self.channels_7x7, 1, 7)
        self.branch7x7dbl_4 = BasicConv2d(self.channels_7x7, 7, 1)
        self.branch7x7dbl_5 = BasicConv2d(192, 1, 7)

        self.avgpool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')
        self.branch_pool = BasicConv2d(192, 1, 1)

    def __call__(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = self.avgpool(x)
        branch_pool = self.branch_pool(branch_pool)

        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=self.channel_axis,
            name=self.name)

        return x


class InceptionD(object):
    def __init__(self, in_channels, name):
        self.in_channels = in_channels
        self.name = name

        if K.image_data_format() == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = 3

        self.branch3x3_1 = BasicConv2d(192, 1, 1)
        self.branch3x3_2 = BasicConv2d(320, 3, 3, strides=(2, 2), padding='valid')

        self.branch7x7x3_1 = BasicConv2d(192, 1, 1)
        self.branch7x7x3_2 = BasicConv2d(192, 1, 7)
        self.branch7x7x3_3 = BasicConv2d(192, 7, 1)
        self.branch7x7x3_4 = BasicConv2d(192, 3, 3, strides=(2, 2), padding='valid')

        self.maxpool = MaxPooling2D((3, 3), strides=(2, 2))

    def __call__(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = self.maxpool(x)
        x = layers.concatenate(
            [branch3x3, branch7x7x3, branch_pool], axis=self.channel_axis, name=self.name)

        return x


class InceptionE(object):
    def __init__(self, in_channels, name_idx):
        self.in_channels = in_channels
        self.name_idx = name_idx

        if K.image_data_format() == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = 3

        self.branch1x1 = BasicConv2d(320, 1, 1)

        self.branch3x3_1 = BasicConv2d(384, 1, 1)
        self.branch3x3_2a = BasicConv2d(384, 1, 3)
        self.branch3x3_2b = BasicConv2d(384, 3, 1)

        self.branch3x3dbl_1 = BasicConv2d(448, 1, 1)
        self.branch3x3dbl_2 = BasicConv2d(384, 3, 3)
        self.branch3x3dbl_3a = BasicConv2d(384, 1, 3)
        self.branch3x3dbl_3b = BasicConv2d(384, 3, 1)

        self.avgpool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')
        self.branch_pool = BasicConv2d(192, 1, 1)

    def __call__(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3_1 = self.branch3x3_2a(branch3x3)
        branch3x3_2 = self.branch3x3_2b(branch3x3)

        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=self.channel_axis, name='mixed9_' + str(self.name_idx))  # 'mixed9_' + str(i)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl_1 = self.branch3x3dbl_3a(branch3x3dbl)
        branch3x3dbl_2 = self.branch3x3dbl_3b(branch3x3dbl)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=self.channel_axis)

        branch_pool = self.avgpool(x)
        branch_pool = self.branch_pool(branch_pool)

        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=self.channel_axis,
            name='mixed' + str(9 + self.name_idx))  # 'mixed' + str(9 + i)

        return x


class InceptionV3(object):
    def __init__(self, include_top=True, classes=1000, pooling=None):
        self.include_top = include_top
        self.pooling = pooling

        self.Conv2d_1a_3x3 = BasicConv2d(32, 3, 3, strides=(2, 2), padding='valid')
        self.Conv2d_2a_3x3 = BasicConv2d(32, 3, 3, padding='valid')
        self.Conv2d_2b_3x3 = BasicConv2d(64, 3, 3)
        self.maxpool1 = MaxPooling2D((3, 3), strides=(2, 2))
        self.Conv2d_3b_1x1 = BasicConv2d(80, 1, 1, padding='valid')
        self.Conv2d_4a_3x3 = BasicConv2d(192, 3, 3, padding='valid')
        self.maxpool2 = MaxPooling2D((3, 3), strides=(2, 2))
        self.Mixed_5b = InceptionA(192, pool_features=32, name='mixed0', branch_pool_filters=32)
        self.Mixed_5c = InceptionA(256, pool_features=64, name='mixed1', branch_pool_filters=64)
        self.Mixed_5d = InceptionA(288, pool_features=64, name='mixed2', branch_pool_filters=64)
        self.Mixed_6a = InceptionB(288, name='mixed3')
        self.Mixed_6b = InceptionC(768, channels_7x7=128, name='mixed4')
        self.Mixed_6c = InceptionC(768, channels_7x7=160, name='mixed5')
        self.Mixed_6d = InceptionC(768, channels_7x7=160, name='mixed6')
        self.Mixed_6e = InceptionC(768, channels_7x7=192, name='mixed7')

        self.Mixed_7a = InceptionD(768, name='mixed8')
        self.Mixed_7b = InceptionE(1280, name_idx=0)
        self.Mixed_7c = InceptionE(2048, name_idx=1)

        self.classifier_GAP = GlobalAveragePooling2D(name='avg_pool')
        self.classifier_fc = Dense(classes, activation='softmax', name='predictions')

        self.GAP = GlobalAveragePooling2D()
        self.GMP = GlobalMaxPooling2D()

    def __call__(self, img_input):
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(img_input)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = self.maxpool1(x)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = self.maxpool2(x)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)

        if self.include_top:
            x = self.classifier_GAP(x)
            x = self.classifier_fc(x)
        else:
            if self.pooling == 'avg':
                x = self.GAP(x)
            elif self.pooling == 'max':
                x = self.GMP(x)

        return x
