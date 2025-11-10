import os
import cv2
import torch
import numpy as np
from tqdm import tqdm


def load_image_gray(path):
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def preprocess(img):
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # [B=1, C=1, H, W]
    return img


def postprocess(mask):
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask


def inference_single(model, img_path, device):
    img = load_image_gray(img_path)
    h, w = img.shape

    inp = preprocess(img).to(device)

    with torch.no_grad():
        pred = model(inp)
        pred = pred.squeeze().cpu().numpy()

    mask = postprocess(pred)
    return mask, (h, w)


def inference(
        model_class,
        encoder,
        checkpoint_path,
        input_path,
        output_dir="outputs",
        save_result=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====== 加载模型 ======
    model = model_class(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=1,
        classes=1,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # ====== 处理输入路径 ======
    if os.path.isdir(input_path):
        img_list = [os.path.join(input_path, f) for f in os.listdir(input_path)]
    else:
        img_list = [input_path]

    if save_result:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- Running Inference on {len(img_list)} images ---")

    results = []
    for img_path in tqdm(img_list, ncols=100):
        mask, _ = inference_single(model, img_path, device)
        results.append((img_path, mask))

        if save_result:
            name = os.path.basename(img_path)
            save_path = os.path.join(output_dir, f"{name}_mask.png")
            cv2.imwrite(save_path, mask)

    print("\nInference Completed.")
    return results
