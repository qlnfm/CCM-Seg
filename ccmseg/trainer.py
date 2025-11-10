import os
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from .loss import SoftCLDiceLoss
from .loader import Dataset

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"


class Trainer:
    def __init__(
            self,
            model: smp.base.SegmentationModel,
            encoder: str,
    ):
        self.model = model(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=1,
            classes=1
        )

    def _create_loader(self, root, name, images_dir_name, masks_dir_name, batch_size, aug, shuffle):
        dataset = Dataset(
            os.path.join(root, name),
            images_dir_name,
            masks_dir_name,
            augmentation=aug
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    def _run_epoch(self, model, loader, loss_fn, device, train=False, optimizer=None, desc=""):
        model.train() if train else model.eval()

        epoch_loss = 0.0

        # 带进度条
        progress = tqdm(loader, desc=desc, ncols=100, leave=False)

        for images, masks in progress:
            images, masks = images.to(device), masks.to(device)

            with torch.set_grad_enabled(train):
                outputs = model(images)
                loss = loss_fn(outputs, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

            # 更新进度条显示当前 batch loss
            progress.set_postfix(loss=f"{loss.item():.4f}")

        return epoch_loss / len(loader)

    def train(
            self,
            data_dir,
            data_names,
            image_dir_name='images',
            mask_dir_name='masks',
            batch_size=4,
            augmentation=True,
            num_epochs=250,
            early_stop=3,
            lr=1e-3,
            use_softcdiceloss=False,
            save_path='best_model.pth',
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ====== DataLoaders ======
        train_loader = self._create_loader(data_dir, data_names[0], image_dir_name, mask_dir_name,
                                           batch_size, augmentation, shuffle=True)
        val_loader = self._create_loader(data_dir, data_names[1], image_dir_name, mask_dir_name,
                                         batch_size, False, shuffle=False)
        test_loader = self._create_loader(data_dir, data_names[2], image_dir_name, mask_dir_name,
                                          batch_size, False, shuffle=False)

        print('Data loading is complete.')

        # ====== Model & Loss ======
        model = self.model.to(device)

        loss_fn = SoftCLDiceLoss(alpha=0.5, region='dice', skel_iter=6).to(device) \
            if use_softcdiceloss else smp.losses.DiceLoss(mode='binary')

        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float('inf')
        no_improve = 0

        print("\n--- Start Training ---")
        for epoch in range(num_epochs):
            train_loss = self._run_epoch(
                model, train_loader, loss_fn, device,
                train=True, optimizer=optimizer,
                desc=f"Epoch {epoch + 1}/{num_epochs} Train"
            )

            val_loss = self._run_epoch(
                model, val_loader, loss_fn, device,
                train=False,
                desc=f"Epoch {epoch + 1}/{num_epochs} Val"
            )

            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                print(f"Validation improved {best_val_loss:.4f} → {val_loss:.4f}, saving...")
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= early_stop:
                print("\nEarly stop triggered.")
                break

        print("\nTraining finished.")

        # ====== Final test ======
        print("--- Starting final evaluation on Test Set ---")
        model.load_state_dict(torch.load(save_path))
        model.eval()

        print(f"Final model loaded from '{save_path}'.")

        test_loss = self._run_epoch(
            model, test_loader, loss_fn, device,
            train=False,
            desc="Testing"
        )

        print(f"Test Dice: {(1 - test_loss):.4f}")

        return test_loss
