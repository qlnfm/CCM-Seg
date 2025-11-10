import os
import cv2
import numpy as np
import torch as th
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    def __init__(
            self,
            ccm_dir,
            images_dir_name='images',
            masks_dir_name='masks',
            augmentation=False
    ):
        self.images_dir = os.path.join(ccm_dir, images_dir_name)
        self.masks_dir = os.path.join(ccm_dir, masks_dir_name)

        self.images_fps = sorted([
            os.path.join(self.images_dir, f) for f in os.listdir(self.images_dir)
        ])
        self.masks_fps = sorted([
            os.path.join(self.masks_dir, f) for f in os.listdir(self.masks_dir)
        ])

        self.augmentation = augmentation

    def _load_image(self, path):
        with open(path, "rb") as f:
            buf = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

    def _apply_transform(self, img, mask, t):
        if t == 0:
            return img, mask

        # t in [0,7]: 4 rotations × 2 flips
        flip = (t & 1) == 1
        rot = t >> 1

        if flip:
            img = np.fliplr(img)
            mask = np.fliplr(mask)

        if rot:
            img = np.rot90(img, rot)
            mask = np.rot90(mask, rot)

        return img, mask

    def __getitem__(self, i):
        if self.augmentation:
            orig_i = i // 8
            t = i % 8
        else:
            orig_i = i
            t = 0

        image = self._load_image(self.images_fps[orig_i])
        mask = self._load_image(self.masks_fps[orig_i])

        image, mask = self._apply_transform(image, mask, t)

        image = (image.astype(np.float32) / 255.0)
        mask = (mask > 0).astype(np.float32)

        return th.from_numpy(image).unsqueeze(0), th.from_numpy(mask).unsqueeze(0)

    def __len__(self):
        return len(self.images_fps) * 8 if self.augmentation else len(self.images_fps)


if __name__ == '__main__':
    dataset_no_aug = Dataset('./data/ccm')
    dataset_with_aug = Dataset('./data/ccm', augmentation=True)

    print(f"Dataset without augmentation has {len(dataset_no_aug)} samples.")
    print(f"Dataset with augmentation has {len(dataset_with_aug)} samples.")
    print("DataLoader is ready.")
