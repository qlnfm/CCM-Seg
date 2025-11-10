from ccmseg import train_segmentation_model

train_segmentation_model(
    'data',
    'set1 - train',
    'set1 - valid',
    'set2',
    'images',
    'annotations',
    batch_size=32,
    save_path='output/model1.pth',
)
