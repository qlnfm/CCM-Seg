import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split


class SegmentationDataset(Dataset):
    def __init__(self, files, img_dir, mask_dir, transform=None):
        self.files = files
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img = Image.open(os.path.join(self.img_dir, name)).convert("L")
        mask = Image.open(os.path.join(self.mask_dir, name)).convert("L")

        img = self.transform(img)

        mask = torch.from_numpy(np.array(mask, dtype=np.float32) / 255.0)
        mask = mask.unsqueeze(0)

        return img, mask


class DInterface(pl.LightningDataModule):
    def __init__(
            self,
            data_dir,
            images_dirname='images', masks_dirname='annotations',
            set_id=1,
            batch_size=16,
            val_ratio=0.1,
            random_state=42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, images_dirname)
        self.mask_dir = os.path.join(data_dir, masks_dirname)
        self.set_id = str(set_id)
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.random_state = random_state

    def setup(self, stage=None):
        files = [f for f in os.listdir(self.img_dir) if f.endswith(".png") and f.startswith(self.set_id)]
        participants = sorted({f[1:3] for f in files})
        train_pp, val_pp = train_test_split(participants, test_size=self.val_ratio, random_state=self.random_state)

        split = lambda pp: [f for f in files if f[1:3] in pp]
        self.train_ds = SegmentationDataset(split(train_pp), self.img_dir, self.mask_dir)
        self.val_ds = SegmentationDataset(split(val_pp), self.img_dir, self.mask_dir)

    def _loader(self, ds, shuffle=False):
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds)


if __name__ == '__main__':
    d = DInterface(data_dir='../Dataset', set_id=1, batch_size=8)
    print(d)
