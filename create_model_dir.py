import os


def create_model_dir(args):
    model_dir = args.experiment_dir + '/models/' + args.use_new_network
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print("Model directory has been created:", model_dir)
    return model_dir
