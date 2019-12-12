# Dogs_vs_Cats_TensorFlow_Keras
`Using Keras(TensorFlow) for Dogs vs, Cats match.`

# Introduction
- This repository is for kaggle `Dogs vs. Cats` match, but you can utilize this code to learn how to use `keras`. 
- For network, I has estabilished the structure containing the introduction of pre-trained models like `VGG` , `InceptionV3` and `ResNet`.
- For lr_scheduler, I has designed a multistep lr_scheduler to adjust the learning rate for the optimizer.
- For optimizer, only `Adam` and `SGD` are illustrated in my repository.

# Environment
- Python 3.6
- tensorflow-gpu 1.8.0
- tensorboard 1.8.0

# Dataset Structure
## Original
```
${project_dir}/datasets
    dogsvscats
        train.zip
        test1.zip
```
## Extract train and test datasets
After downloading the datasets from Kaggle website, you need to extract these two zips.(Actually, I just extract train.zip)
```
${project_dir}/datasets
    dogsvscats
        train.zip
        test1.zip
        train           # Extracted from train.zip
        test1           # Extracted from test1.zip
```
## Final step
- You need to run `split_dataset.py` in `dataset_utils` directory to split the train dataset into a new one.
```
${project_dir}/datasets
    dogsvscats
        train.zip
        test1.zip
        train           # Extracted from train.zip
        test1           # Extracted from test1.zip
        cats_and_dogs_large
            test        # For test
            train       # For train
            validation  # For validation
```
# Training
- ResidualNet: `python demo.py --use_network RESNET50 --use_new_network ResidualNet --only_test False`.
- VGGNet: `python demo.py --use_network VGG16 --use_new_network VGGNet --only_test False `.
- InceptionNet: `python demo.py --use_network INCEPTIONV3 --use_new_network InceptionNet --only_test False`.

# Test
- Just `--only_test True`.

# Experimental Directory Structure
- Before training, you need to modify the directories in `parser_setting.py`
- Run `demo.py` to start the training process. The follow directories will be created automatically.
```
${weight_results_dir}
    Dogs_vs_Cats_TensorFlow
        models
            InceptionNet
                best.h5
            ResidualNet
                best.h5
            VGGNet
                best.h5
        pretrained_models
            inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
            resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
            vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
        tensorboard
            ...
```
# Performances
| Network | Accuracy(%)|
|---|---|
| VGGNet | 92.72 |
| ResidualNet | 98.88 |
| InceptionNet | 99.44 |
- I has just trained the models for 10 epochs by 'Adam'.

# Attention
- I has modified the pretrained model code to let those architectures more like that in Pytorch.
- For example, the framework of ResNet50 coded by Keras is modified into a new style, which is the same in Pytorch.
```
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import AveragePooling2D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import ZeroPadding2D

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers


class conv_block(object):
    def __init__(self, kernel_size, filters, stage, block, strides=(2, 2)):
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        self.conv1 = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')
        self.bn1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')
        self.conv2 = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')
        self.bn2 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')
        self.conv3 = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')
        self.bn3 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')
        self.shortcut_conv = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')
        self.shortcut_bn = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')
        self.relu = Activation('relu')

    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        shortcut = self.shortcut_conv(input_tensor)
        shortcut = self.shortcut_bn(shortcut)

        x = layers.add([x, shortcut])
        x = self.relu(x)
        return x


class identity_block(object):
    def __init__(self, kernel_size, filters, stage, block):
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        self.conv1 = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')
        self.bn1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')
        self.conv2 = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')
        self.bn2 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')
        self.conv3 = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')
        self.bn3 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')
        self.relu = Activation('relu')

    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = layers.add([x, input_tensor])
        x = self.relu(x)
        return x


class ResNet50(object):
    def __init__(self, include_top=True, classes=1000, pooling=None):
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        self.include_top = include_top
        self.pooling = pooling

        self.conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')
        self.bn1 = BatchNormalization(axis=bn_axis, name='bn_conv1')
        self.relu = Activation('relu')
        self.maxpool = MaxPooling2D((3, 3), strides=(2, 2))
        self.layer1_list = [
            conv_block(3, [64, 64, 256], stage=2, block='a', strides=(1, 1)),
            identity_block(3, [64, 64, 256], stage=2, block='b'),
            identity_block(3, [64, 64, 256], stage=2, block='c'),
        ]
        self.layer2_list = [
            conv_block(3, [128, 128, 512], stage=3, block='a'),
            identity_block(3, [128, 128, 512], stage=3, block='b'),
            identity_block(3, [128, 128, 512], stage=3, block='c'),
            identity_block(3, [128, 128, 512], stage=3, block='d'),
        ]
        self.layer3_list = [
            conv_block(3, [256, 256, 1024], stage=4, block='a'),
            identity_block(3, [256, 256, 1024], stage=4, block='b'),
            identity_block(3, [256, 256, 1024], stage=4, block='c'),
            identity_block(3, [256, 256, 1024], stage=4, block='d'),
            identity_block(3, [256, 256, 1024], stage=4, block='e'),
            identity_block(3, [256, 256, 1024], stage=4, block='f'),
        ]
        self.layer4_list = [
            conv_block(3, [512, 512, 2048], stage=5, block='a'),
            identity_block(3, [512, 512, 2048], stage=5, block='b'),
            identity_block(3, [512, 512, 2048], stage=5, block='c'),
        ]
        self.avgpool = AveragePooling2D((7, 7), name='avg_pool')
        self.flatten = Flatten()
        self.fc = Dense(classes, activation='softmax', name='fc1000')

        self.GAP = GlobalAveragePooling2D()
        self.GMP = GlobalMaxPooling2D()

    def layer1(self, x):
        for i in range(len(self.layer1_list)):
            x = self.layer1_list[i](x)
        return x

    def layer2(self, x):
        for i in range(len(self.layer2_list)):
            x = self.layer2_list[i](x)
        return x

    def layer3(self, x):
        for i in range(len(self.layer3_list)):
            x = self.layer3_list[i](x)
        return x

    def layer4(self, x):
        for i in range(len(self.layer4_list)):
            x = self.layer4_list[i](x)
        return x

    def __call__(self, img_input):
        x = self.conv1(img_input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        if self.include_top:
            x = self.flatten(x)
            x = self.fc(x)
        else:
            if self.pooling == 'avg':
                x = self.GAP(x)
            elif self.pooling == 'max':
                x = self.GMP(x)
        return x

```
