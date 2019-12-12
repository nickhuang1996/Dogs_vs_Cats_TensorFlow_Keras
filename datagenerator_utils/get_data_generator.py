from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


def get_data_generator(args):
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        directory=args.train_dir,
        target_size=args.CenterCropSize,
        batch_size=args.batch_size,
        class_mode='categorical'
    )
    test_generator = test_datagen.flow_from_directory(
        directory=args.test_dir,
        target_size=args.CenterCropSize,
        batch_size=args.batch_size,
        class_mode='categorical'
    )

    return train_generator, test_generator
