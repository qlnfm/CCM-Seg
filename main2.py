from ccmseg import train_segmentation_model

train_segmentation_model(
    'data',
    'set2 - train',
    'set2 - valid',
    'set1',
    'images',
    'annotations',
    batch_size=32,
    save_path='output/model2.pth',
)
