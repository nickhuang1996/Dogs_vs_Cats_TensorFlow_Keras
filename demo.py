from parse_setting import args

from create_model_dir import create_model_dir
from networks import get_model
from optimizer import get_optimizer
from datagenerator_utils.get_data_generator import get_data_generator
from callbacks import get_callbacks
from tensorflow.python.keras.models import load_model

if __name__ == '__main__':
    model_dir = create_model_dir(args=args)
    args.model_dir = model_dir
    if args.only_test is True:
        model = load_model(args.model_dir + '/best.h5')
        print("Test model has been loaded:", args.model_dir + '/best.h5 ...')
    else:
        if args.resume is True:
            model = load_model(args.model_dir + '/best.h5')
            print("Last best model has been loaded:", args.model_dir + '/best.h5 ...')
        else:
            model = get_model(args=args)
            print("Model has been initialized ...")

    optimizer = get_optimizer(args=args)
    model.compile(loss=args.use_loss, optimizer=optimizer, metrics=['acc'])
    print(model.summary())
    if args.only_test is not True:
        train_generator, test_generator = get_data_generator(args=args)
        callbacks = get_callbacks(args=args)
        print("Start training process..")
        model.fit_generator(train_generator,
                            steps_per_epoch=train_generator.samples // args.batch_size,
                            validation_data=test_generator,
                            validation_steps=test_generator.samples // args.batch_size,
                            workers=args.num_workers,
                            callbacks=callbacks,
                            epochs=args.epochs,
                            )
        model.save_weights(args.model_dir + '/final-weights.h5')
        model.save(args.model_dir + '/final.h5')
    else:
        _, test_generator = get_data_generator(args=args)
        pred = model.evaluate_generator(test_generator,
                                        workers=args.num_workers)
        print("val_loss: {:.4f} val_acc: {:.4f}".format(pred[0], pred[1]))

