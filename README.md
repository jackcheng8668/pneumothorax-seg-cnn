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

The only preprocessing that was done was histogram normalization and rescaling the intensities to [0, 255].

## Baselines

### [1] EfficientNet + UEfficientNet
[0.7976 Public LB]
The first baseline is a classification + segmentation cascade.
* The classification stage is the EfficientNetB4 architecture with the same settings as the kernel below.
  * Uses binary cross entropy & accuracy and F1-score
* The segmentation stage comes from [this kernel](https://www.kaggle.com/meaninglesslives/unet-with-efficientnet-encoder-in-keras), which uses [albumentations](https://github.com/albu/albumentations). The data augmentation is not properly optimized.
  * However, I changed the residual blocks so that it's LeakyReLU -> BN
  instead of BN -> LeakyReLU.
  * uses binary cross entropy + dice loss & mean iou for evaluation
The baseline uses the SnapshotCallbackBuilder and Stochastic Weight Averaging.
* Weights used:
  * Classifier: [b4_pneumothorax_ckpoint_epoch15.model](https://drive.google.com/open?id=1P-o6CNz0nJDaDhg4Djo0Tz-f_I5RCSWn) [0.7886 Public LB]
  * Segmentation Model: [uefficientnet_pneumothorax_pos_only_epoch70_checkpoint.model](https://drive.google.com/open?id=1w9k6WzSjIue51FE6q7zgmCtWR2gMJ-w3)

### [2] From the RSNA Pneumonia Detection 1st Place Solution
[In-Progress]
The second baseline is a classification + segmentation cascade.
* The classification stage comes from [the 1st place solution for the 2018 RSNA Pneumonia Detection Challenge](https://github.com/i-pan/kaggle-rsna18). Their full description is located [here](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/discussion/70421).
  * Fine-tunes ImageNet pretrained InceptionResNetV2, Xception, and DenseNet169 that are subsequently pretrained on the NIH ChestXray14 dataset.
* The segmentation stage is similar to their detection ensemble, except that I change up the architectures
  * 5-fold UResNet
  * 5-fold UEfficientNet
  * 5-fold Attention-Gated U-Net?
  * idk

## EfficientNet-like Training
Next I'm going to try to create 3 levels of architectures based on the U-Net and [the EfficientNet](https://arxiv.org/abs/1905.11946)
way of training (d, w, r). As such, I need to create the GridSearch aspect of the pipeline and create a reasonable B0.
