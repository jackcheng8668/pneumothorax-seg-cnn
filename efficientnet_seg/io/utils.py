import numpy as np

def scale(x, model_name=None):
    # scaling input to [0, 1]
    return (x-x.min()) / (x.max() - x.min())

def zscore_standardize(x, model_name=None):
    return (x-x.mean()) / x.std()

def standardize(x, model_name, mean=0.529, std=0.259):
    return (x-mean) / std

def preprocess_input(x, model_name):
    """
    Preprocess some numpy array input, x, in the style of the user-specified model_name.
    Supports both grayscale and RGB inputs. Assumes channels_last.
    Args:
        x (np.ndarray): (x, y, z, n_channels)
        model_name (str): Either `inception`, `xception`, `mobilenet`, `resnet`, `vgg`, or `densenet`.
            Anything other than those strings will result in the array just being converted to a float.
    """
    x = x.astype("float32")
    if isinstance(model_name, str):
        if model_name in ("inception","xception","mobilenet"):
            x /= 255.
            x -= 0.5
            x *= 2.
        if model_name in ("densenet"):
            x /= 255.
            if x.shape[-1] == 3:
                x[..., 0] -= 0.485
                x[..., 1] -= 0.456
                x[..., 2] -= 0.406
                x[..., 0] /= 0.229
                x[..., 1] /= 0.224
                x[..., 2] /= 0.225
            elif x.shape[-1] == 1:
                x[..., 0] -= 0.449
                x[..., 0] /= 0.226
        elif model_name in ("resnet","vgg"):
            if x.shape[-1] == 3:
                x[..., 0] -= 103.939
                x[..., 1] -= 116.779
                x[..., 2] -= 123.680
            elif x.shape[-1] == 1:
                x[..., 0] -= 115.799
    return x
