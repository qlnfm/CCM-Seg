# CCM-Seg：Quickly train your CCM segmentation model!

## Download Dataset

`
https://doi.org/10.5281/zenodo.17570503
`

## Prepare Dataset

Download the dataset and decompress it, then place it in the root directory.

```
Dataset
 - annotations
 - images
```

## Train
```shell
train.py
```
### Hyperparameter
```text
    DATA_DIR = "./Dataset"
    SET_ID = 2  # 1 or 2 for train
    BATCH_SIZE = 8
    EPOCHS = 20
    LR = 1e-3
    SAVE_DIR = f"checkpoints_set{SET_ID}"
```

## Predict
```shell
predict.py
```
### Hyperparameter
```text
    CKPT_PATH = "checkpoints_set2/segmentation/version_4/checkpoints/best-epoch=18-val_loss=0.1213.ckpt"
    INPUT_PATH = "./Dataset/images"
    SET_ID = 1  # 1 or 2 for predict
    OUT_DIR = f"./Output/predictions_set{SET_ID}/{CKPT_PATH.split('/')[2]}"
```
