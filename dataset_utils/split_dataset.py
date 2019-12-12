import os
import shutil

TRAIN_LENGTH = 11250
VALIDATION_LENGTH = 625
TEST_LENGTH = 625

# dataset after being extracted
original_dataset_dir = 'D:/datasets/dogsvscats/train'
# New directory for training and test
base_dir = 'D:/datasets/dogsvscats/cats_and_dogs_large'
os.mkdir(base_dir)

# Build train, validation and test directories
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# The images of dogs and cats will be copied into these three directories
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

# split the dataset
fnames = ['cat.{}.jpg'.format(i) for i in range(TRAIN_LENGTH)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dat = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dat)

fnames = ['cat.{}.jpg'.format(i) for i in range(TRAIN_LENGTH, TRAIN_LENGTH + VALIDATION_LENGTH)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dat = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dat)

fnames = ['cat.{}.jpg'.format(i) for i in range(TRAIN_LENGTH + VALIDATION_LENGTH, TRAIN_LENGTH + VALIDATION_LENGTH + TEST_LENGTH)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dat = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dat)

fnames = ['dog.{}.jpg'.format(i) for i in range(TRAIN_LENGTH)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dat = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dat)

fnames = ['dog.{}.jpg'.format(i) for i in range(TRAIN_LENGTH, TRAIN_LENGTH + VALIDATION_LENGTH)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dat = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dat)

fnames = ['dog.{}.jpg'.format(i) for i in range(TRAIN_LENGTH + VALIDATION_LENGTH, TRAIN_LENGTH + VALIDATION_LENGTH + TEST_LENGTH)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dat = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dat)
