import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from seg.model import MInterface


def load_model(ckpt_path, device):
    model = MInterface.load_from_checkpoint(ckpt_path)
    model.eval()
    model.to(device)
    return model


def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert("L")
    tensor = transform(img).unsqueeze(0)  # (1,1,H,W)
    return tensor


def save_mask(mask_tensor, save_path):
    mask = mask_tensor.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    Image.fromarray(mask).save(save_path)


@torch.no_grad()
def predict_single(model, image_path, save_path, device):
    x = preprocess(image_path)
    x = x.to(device)

    logits = model(x)
    probs = torch.sigmoid(logits)

    save_mask(probs, save_path)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(CKPT_PATH, device)

    os.makedirs(OUT_DIR, exist_ok=True)

    if os.path.isfile(INPUT_PATH):
        filename = os.path.basename(INPUT_PATH)
        save_path = os.path.join(OUT_DIR, filename)
        predict_single(model, INPUT_PATH, save_path, device)

    else:
        files = [f for f in os.listdir(INPUT_PATH) if f.endswith(".png") and f.startswith(str(SET_ID))]
        for f in files:
            img_path = os.path.join(INPUT_PATH, f)
            save_path = os.path.join(OUT_DIR, f)
            predict_single(model, img_path, save_path, device)

    print("Done.")


if __name__ == "__main__":
    CKPT_PATH = "checkpoints_set2/segmentation/version_4/checkpoints/best-epoch=18-val_loss=0.1213.ckpt"
    INPUT_PATH = "./Dataset/images"
    SET_ID = 1  # 1 or 2 for predict
    OUT_DIR = f"./Output/predictions_set{SET_ID}/{CKPT_PATH.split('/')[2]}"

    main()
