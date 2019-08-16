from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

from efficientnet_seg.training.callbacks import SnapshotCallbackBuilder, SWA
from efficientnet_seg.models.losses_metrics import f1

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
