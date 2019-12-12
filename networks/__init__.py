from tensorflow.python.keras.models import Model
from networks.ResidualNet import ResidualNet
from networks.VGGNet import VGGNet
from networks.InceptionNet import InceptionNet
FREEZE_LAYERS = 0


def get_model(args):
    network = eval(args.use_new_network)(args=args)
    output = network()
    model = Model(inputs=network.features.input, outputs=output)

    for layer in model.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in model.layers[FREEZE_LAYERS:]:
        layer.trainable = True

    return model

