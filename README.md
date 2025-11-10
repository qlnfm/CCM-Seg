# CCM-Seg：Quickly train your CCM segmentation model!

## Download Dataset

`
https://zenodo.org/records/17570503
`

## Prepare Dataset

Organize the folder in the following format:

```
data
 - train
    - images
    - annotations
 - valid
    - images
    - annotations
 - test
    - images
    - annotations
```

## Start Training
```python
from ccmseg import train_segmentation_model

train_segmentation_model(
    'data',
    'train',
    'valid',
    'test',
    'images',
    'annotations',
    batch_size=32,
    save_path='best.pth',
)
```

## Inference
```python
from ccmseg import run_inference

run_inference(
    'best.pth',
    'data/test/images',
    output_dir='infer'
)
```
