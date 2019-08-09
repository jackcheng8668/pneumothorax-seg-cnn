from tensorflow.keras.layers import Add, Concatenate, MaxPooling2D, \
                                    UpSampling2D, LeakyReLU, \
                                    Conv2D, BatchNormalization
from efficientnet_seg.models.unet.instance_norm import InstanceNormalization
from tensorflow.keras import backend as K

def localization_module_2D(input_layer, skip_layer, n_filters, upsampling_size=(2,2), n_convs=2, instance_norm=True):
    """
    [2D]; Channels_first
    Localization module (Downsampling compartment of the U-Net): UpSampling2D -> `n_convs` Convs (w/ LeakyReLU and BN)
    Args:
        input_layer (tf.keras layer):
        skip_layer (tf.keras layer): layer with the corresponding skip connection (same depth)
        n_filters (int): number of filters for each conv layer
        upsampling_size (tuple):
        n_convs (int): Number of convolutions in the module
        instance_norm (bool): whether or not to use instance normalization
    Returns:
        upsampled output
    """
    data_format = K.image_data_format()
    upsamp = UpSampling2D(upsampling_size)(input_layer)
    concat = Concatenate(axis=1 if data_format == "channels_first" else -1)([upsamp, skip_layer])
    return context_module_2D(concat, n_filters=n_filters, pool_size=None, n_convs=n_convs, instance_norm=instance_norm)

def context_module_2D(input_layer, n_filters, pool_size=(2,2), n_convs=2, instance_norm=True):
    """
    [2D]
    Context module (Downsampling compartment of the U-Net): `n_convs` Convs (w/ LeakyReLU and BN) -> MaxPooling
    Args:
        input_layer (tf.keras layer):
        n_filters (int): number of filters for each conv layer
        pool_size (tuple):
        n_convs (int): Number of convolutions in the module
        instance_norm (bool): whether or not to use instance normalization
    Returns:
        keras layer after double convs w/ LeakyReLU and BN in-between
        maxpooled output
    """
    data_format = K.image_data_format()
    for conv_idx in range(n_convs):
        if conv_idx == 0:
            conv = Conv2D(n_filters, kernel_size=(3,3), padding='same')(input_layer)
        else:
            conv = Conv2D(n_filters, kernel_size=(3,3), padding='same')(bn)
        act = LeakyReLU(0.3)(conv)
        if instance_norm:
            bn = InstanceNormalization(axis=1 if data_format == "channels_first" else -1)(act)
        else:
            bn = BatchNormalization(axis=1 if data_format == "channels_first" else -1)(act)
    if pool_size is not None:
        pool = MaxPooling2D(pool_size)(bn)
        return bn, pool
    elif pool_size is None:
        return bn
