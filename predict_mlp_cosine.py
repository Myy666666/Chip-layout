import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms

from hybrid_cnn_dino import DEFAULT_SUPERPOINT_PATH, HybridCnnDinoNet
from mlp_head import MLPHead
from train_mlp_cosine import IMG_SIZE, pool_feature_for_mlp


ROOT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT_DIR / "weights"

DEFAULT_DINO_CKPT = Path(os.environ.get("DINO_CKPT_PATH", WEIGHTS_DIR / "dinov2_vitl14_pretrain.pth"))
DEFAULT_MLP_WEIGHT = Path(os.environ.get("MLP_WEIGHT_PATH", WEIGHTS_DIR / "mlp_head_triplet-0410.pth"))
DEFAULT_SUPERPOINT_WEIGHT = Path(os.environ.get("SUPERPOINT_WEIGHT_PATH", DEFAULT_SUPERPOINT_PATH))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def normalize_state_dict(state):
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("MLP checkpoint is not a valid state_dict.")

    return {key.replace("module.", ""): value for key, value in state.items()}


def infer_mlp_shape(state_dict):
    if "mlp.8.weight" in state_dict:
        return (
            state_dict["mlp.0.weight"].shape[0],
            state_dict["mlp.4.weight"].shape[0],
            state_dict["mlp.8.weight"].shape[0],
        )
    if "mlp.4.weight" in state_dict:
        return (
            state_dict["mlp.0.weight"].shape[0],
            None,
            state_dict["mlp.4.weight"].shape[0],
        )
    raise ValueError("Cannot infer MLP structure from checkpoint.")


def load_models(dino_ckpt_path=DEFAULT_DINO_CKPT, mlp_weight_path=DEFAULT_MLP_WEIGHT):
    backbone = HybridCnnDinoNet(
        dino_ckpt_path=dino_ckpt_path,
        dino_feature_layer=[23],
        superpoint_weight_path=DEFAULT_SUPERPOINT_WEIGHT,
        load_ckpt=True,
    ).to(DEVICE)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    with torch.no_grad():
        dummy = torch.zeros(1, 3, *IMG_SIZE).to(DEVICE)
        feat = pool_feature_for_mlp(backbone(dummy))
        feat_dim = feat.shape[1] * feat.shape[2] * feat.shape[3]

    state_dict = normalize_state_dict(torch.load(str(mlp_weight_path), map_location="cpu"))
    hidden_dim, hidden_dim2, out_dim = infer_mlp_shape(state_dict)
    mlp_head = MLPHead(feat_dim, hidden_dim, hidden_dim2, out_dim).to(DEVICE)
    mlp_head.load_state_dict(state_dict, strict=True)
    mlp_head.eval()

    print(
        f"MLP structure: in_dim={feat_dim}, "
        f"hidden_dim={hidden_dim}, hidden_dim2={hidden_dim2}, out_dim={out_dim}"
    )
    return backbone, mlp_head


def get_proj(img_path, backbone, mlp_head, rotate_deg=0):
    img = Image.open(img_path).convert("RGB")
    if rotate_deg:
        img = img.rotate(rotate_deg, expand=True)
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = pool_feature_for_mlp(backbone(img)).reshape(1, -1)
        proj = mlp_head(feat)
    return proj


def show_images(img1_path, img2_path, rotate_deg=0):
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    if rotate_deg:
        img1 = img1.rotate(rotate_deg, expand=True)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title(f"Image 1 rotated {rotate_deg} deg" if rotate_deg else "Image 1")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title("Image 2")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two chip layout images with Hybrid CNN + DINOv2 + MLP.")
    parser.add_argument("img1", help="First image path.")
    parser.add_argument("img2", help="Second image path.")
    parser.add_argument("--rotate-deg", type=float, default=0, help="Rotation angle applied to img1.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Cosine similarity threshold.")
    parser.add_argument("--show", action="store_true", help="Display the two input images before prediction.")
    parser.add_argument("--dino-ckpt", default=str(DEFAULT_DINO_CKPT), help="Path to DINOv2 ViT-L/14 checkpoint.")
    parser.add_argument("--mlp-weight", default=str(DEFAULT_MLP_WEIGHT), help="Path to trained MLP checkpoint.")
    return parser.parse_args()


def main():
    args = parse_args()
    img1_path = Path(args.img1)
    img2_path = Path(args.img2)

    if not img1_path.exists() or not img2_path.exists():
        raise FileNotFoundError(f"Image path not found: {img1_path if not img1_path.exists() else img2_path}")

    if args.show:
        show_images(img1_path, img2_path, rotate_deg=args.rotate_deg)

    backbone, mlp_head = load_models(args.dino_ckpt, args.mlp_weight)
    proj1 = get_proj(img1_path, backbone, mlp_head, rotate_deg=args.rotate_deg)
    proj2 = get_proj(img2_path, backbone, mlp_head)
    cos_sim = torch.nn.functional.cosine_similarity(proj1, proj2).item()

    print(f"Image 1: {img1_path}")
    print(f"Image 2: {img2_path}")
    print(f"Cosine similarity: {cos_sim:.4f}")
    print("Prediction:", "similar" if cos_sim > args.threshold else "not similar")


if __name__ == "__main__":
    main()
