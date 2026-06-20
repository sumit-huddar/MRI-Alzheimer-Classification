# Transfer Learning on MRI Slices — AlexNet & VGG16

Earlier in this repo, transfer learning was only explored on **CIFAR-10**. This notebook brings
that idea to the **ADNI Alzheimer's MRI** data, classifying scans into **AD / CN / MCI**.

## Approach

The 3D-CNN experiments used full volumes. ImageNet-pretrained CNNs are 2D, so here each `.nii`
volume is reduced to a few **central axial slices**, converted to 3-channel `224×224` images, and
fed to two models:

| Model    | Type | Notes |
| :------- | :--- | :---- |
| **VGG16**   | True transfer learning | Frozen ImageNet base + new head, then optional fine-tuning of the last block. |
| **AlexNet** | Trained from scratch | Keras has **no** pretrained AlexNet, so it's the from-scratch baseline (same as the CIFAR notebook). |

## Data

Same as the 3D-CNN notebook — ADNI MRI (Kaggle), NIfTI `.nii`, layout:

```
train/train/{AD, CN, MCI}
val/val/{AD, CN, MCI}
test/test/{AD, CN, MCI}
```

## Pipeline

1. Load each volume with `nibabel`, take `SLICES_PER_VOLUME` central axial slices.
2. Per-slice min-max scale to `0–255`, stack to 3 channels, resize to `224×224`.
3. VGG16 path uses `vgg16.preprocess_input`; AlexNet path scales to `[0, 1]`.
4. Train, then evaluate with shared `plot_history` / `evaluate_predictions` helpers
   (confusion matrix, classification report, OvR ROC-AUC) for comparison with the CIFAR results.

## Tech Stack

Python · TensorFlow / Keras · PyTorch (resize) · Nibabel · NumPy · Matplotlib · scikit-learn · seaborn
