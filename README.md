# EfficientNet for Pneumothorax Segmentation
This repository aims to recreate the training process described in the official EfficientNet paper.

## Preprocessing
The first route was simply to use the data from:
* https://www.kaggle.com/iafoss/data-repack-and-image-statistics
* https://www.kaggle.com/iafoss/siimacr-pneumothorax-segmentation-data-1024
  * mean 0.521, std 0.254
* https://www.kaggle.com/iafoss/siimacr-pneumothorax-segmentation-data-512
  * mean 0.529, std 0.259
* https://www.kaggle.com/iafoss/siimacr-pneumothorax-segmentation-data-256
  * mean 0.540, std 0.264

## Training
* Data is in the `channels_last` format (for the pretrained weights)
### Data Augmentation
Baseline would be [this kernel](https://www.kaggle.com/meaninglesslives/unet-with-efficientnet-encoder-in-keras), which uses [albumentations](https://github.com/albu/albumentations).
Next steps are to try to apply:
* https://github.com/kakaobrain/fast-autoaugment
* https://github.com/arcelien/pba

### Loss Functions & Metrics
Since this part of my pipeline isn't really the focus, I'm going to try to keep it simple
and only use mean iou and BCE dice loss.

### Other Training Settings
* Using the SnapshotCallbackBuilder and SWA as defined in the [baseline kernel](https://www.kaggle.com/meaninglesslives/unet-with-efficientnet-encoder-in-keras).

## Architectures
Baseline would be [this kernel](https://www.kaggle.com/meaninglesslives/unet-with-efficientnet-encoder-in-keras).
  * However, I'm going to change up the residual blocks so that it's LeakyReLU -> BN
  instead of BN -> LeakyReLU.

Next I'm going to try to create 3 levels of architectures based on the U-Net and [the EfficientNet](https://arxiv.org/abs/1905.11946)
way of training (d, w, r). As such, I need to create the GridSearch aspect of the pipeline and
create a reasonable B0.
