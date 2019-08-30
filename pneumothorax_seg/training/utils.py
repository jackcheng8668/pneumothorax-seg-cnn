from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from pneumothorax_seg.training.callbacks import SnapshotCallbackBuilder, SWA
from pneumothorax_seg.models.losses_metrics import f1
from pneumothorax_seg.models.efficientnet.models import EfficientNetB4
from pneumothorax_seg.models.grayscale.densenet import DenseNet169
from pneumothorax_seg.models.grayscale.xception import Xception
from pneumothorax_seg.models.grayscale.inception_resnet_v2 import InceptionResNetV2

from pneumothorax_seg.script_utils.downloaders import download_nih_weights, NIH_WEIGHTS

def load_pretrained_classification_model(model_name="efficientnet", input_shape=None,
                                         dropout=None, pretrained="imagenet"):
    """
    Creates a classification model from pretrained models that are located in this repository.
    Assumes that we are doing a binary classification task with sigmoid (average pooling at the end).
    Args:
        model_name (str): one of `efficientnet` (EfficientNetB4), `densenet` (DenseNet169),
            `inception` (InceptionResNetv2), or `xception` (Xception).
        input_shape (tuple/list): (h, w, n_channels). Defaults to None. If None, then
            we assume efficientnet (256), densenet (448), inception (256), xception (320).
        dropout (float): dropout rate. See `rate` in https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout.
            Defaults to None.
        pretrained (str): one of `None` (random initialization), 'imagenet' (pre-training on ImageNet),
            or `nih` (pre-training on NIH Chest X-Ray14, not available to efficientnet). This is strictly for the weights argument in
            `base_model`.
    Returns:
        A pretrained classification tf.keras.models.Model with the desired input and output shape and loaded weights.
    """
    print("Using {0}...".format(model_name))
    default_models_and_shapes = {"efficientnet": {"base_model": EfficientNetB4, "input_shape": (256, 256, 3)},
                                 "densenet": {"base_model": DenseNet169, "input_shape": (448, 448, 1)},
                                 "inception": {"base_model": InceptionResNetV2, "input_shape": (256, 256, 1)},
                                 "xception": {"base_model": Xception, "input_shape": (320, 320, 1)},
                                }
    model_name = model_name.lower()
    if pretrained == "nih":
        assert model_name != "efficientnet", "NIH pretrained weights are not available for the EfficientNetB4.\
                                              Please change `pretrained` to one of None or `imagenet`"
        # Downloading the NIH pretrained weights
        download_nih_weights(model_name=model_name)
        # setting the path to the pretrained weights
        pretrained = NIH_WEIGHTS[model_name][0]

    # setting up some common reused parameters
    if input_shape is None:
        input_shape = default_models_and_shapes[model_name]["input_shape"]
        print("Using the input shape of {0}".format(input_shape))
    base_model = default_models_and_shapes[model_name]["base_model"]
    # defaults: layer=0, n_classes=1, activation="sigmoid", pooling="avg", weights=None
    model = get_classification_model(base_model, input_shape=input_shape, dropout=dropout,
                                     pretrained=pretrained)
    return model

def get_classification_model(base_model, layer=0, input_shape=(224,224,1), classes=1,
                             activation="sigmoid", dropout=None, pooling="avg",
                             weights=None, pretrained="imagenet"):
    """
    Creates a classification model from a pretrained backbone, such as DenseNet169, InceptionResNetv2.
    From: https://github.com/i-pan/kaggle-rsna18/blob/master/src/train/TrainClassifierEnsemble.py
    Args:
        base_model: function that returns a tf.keras.models.Model, such as tf.keras.applications models.
            This works with any of the classification models in this repository.
        layer (int): max layer index to freeze layers up to. Defaults to 0.
        input_shape (tuple/list): (h, w, n_channels)
        classes (int): number of output classes
        activation (str): output activation function
        dropout (float): dropout rate. See `rate` in https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout.
        pooling (str): either `avg` or `max` to describe the type of pooling to do after outputs of the flattened
            last non-top layer of `base_model`.
        weights (str): Path to weights to load from after the entire model is created.. Defaults to None.
        pretrained (str): one of `None` (random initialization), 'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded. This is strictly for the weights argument in
            `base_model`.
    Returns:
        A remade classification tf.keras.models.Model with the desired input and output shape and weights.
    """
    base = base_model(input_shape=input_shape, include_top=False, weights=pretrained)

    if pooling == "avg":
        x = GlobalAveragePooling2D()(base.output)
    elif pooling == "max":
        x = GlobalMaxPooling2D()(base.output)
    elif pooling is None:
        x = Flatten()(base.output)
    if dropout is not None:
        x = Dropout(dropout)(x)
    x = Dense(classes, activation=activation)(x)
    model = Model(inputs=base.input, outputs=x)
    if weights is not None:
        model.load_weights(weights)
    for l in model.layers[:layer]:
        l.trainable = False
    return model

def compile_classification(model, opt=None, lr=1e-5,):
    """
    Compiling the sparse version of training
    Args:
        model (tf.keras.models.Model): classification model to be compiled
        opt (None or tf.keras.optimizers.Optimizer): Optional. Defaults to None.
        lr (float): learning rate for the optimizer. This LR is overwritten by
            any LR schedule / CLR and is only necessary when opt=None. Defaults to 1e-5.
    """
    if opt is None:
        opt = Adam(lr=lr)
    metrics = [f1, "binary_accuracy"]
    loss = ["binary_crossentropy"]
    model.compile(opt, loss=loss, metrics=metrics)

def get_callbacks_swa_cosine_annealing(monitor="val_loss", mode="min", epochs=30, init_lr=1e-3):
    """
    Quick function to get the callbacks for training.
    Args:
        monitor (str): which metric/loss to monitor for ReduceLROnPlateau and ModelCheckpoint
        mode (str): either "min" or "max" to complement monitor
        epochs: number of training epochs.
        init_lr (float): initial learning rate for the cosine annealing lr schedule.
    """
    csv_logger = CSVLogger("./training_log.csv")
    print("Using SWA and a Cosine Annealing LR Schedule.")
    snapshot = SnapshotCallbackBuilder(nb_epochs=epochs, nb_snapshots=1, init_lr=init_lr)
    swa = SWA("/content/keras_swa.model", epochs-3)
    callbacks_list = snapshot.get_callbacks(swa, monitor, mode) + [csv_logger]
    return callbacks_list

def get_callbacks_reg(monitor="val_loss", mode="min", factor=0.3, patience=2, min_lr=1e-6):
    """
    Quick function to get the callbacks for training.
    Args:
        monitor (str): which metric/loss to monitor for ReduceLROnPlateau and ModelCheckpoint
        mode (str): either "min" or "max" to complement monitor
        factor (float): see tf.keras.callbacks.ReduceLROnPlateau's factor argument at
            https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau
        patience (int): see tf.keras.callbacks.ReduceLROnPlateau's patience argument at
            https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau
        min_lr (float): see tf.keras.callbacks.ReduceLROnPlateau's min_lr argument at
            https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau
    Returns:
        callbacks_list: List of callbacks; contains a CSVLogger callback + ModelCheckpoint + ReduceLROnPlateau
    """
    csv_logger = CSVLogger("./training_log.csv")
    print("Using ModelCheckpoint + ReduceLROnPlateau")
    ckpoint = ModelCheckpoint(filepath="./checkpoint.h5", monitor=monitor, verbose=1, save_best_only=True, mode=mode)
    lr_plat = ReduceLROnPlateau(monitor=monitor, factor=factor, patience=patience, min_lr=min_lr, verbose=1)
    callbacks_list = [csv_logger, ckpoint, lr_plat]
    return callbacks_list
