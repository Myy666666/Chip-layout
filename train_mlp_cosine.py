import os
import random
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from hybrid_cnn_dino import DEFAULT_SUPERPOINT_PATH, HybridCnnDinoNet, set_global_determinism
from mlp_head import MLPHead


ROOT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT_DIR / "weights"
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

CKPT_PATH = Path(os.environ.get("DINO_CKPT_PATH", WEIGHTS_DIR / "dinov2_vitl14_pretrain.pth"))
SUPERPOINT_WEIGHT_PATH = Path(os.environ.get("SUPERPOINT_WEIGHT_PATH", DEFAULT_SUPERPOINT_PATH))
GROUPED_DIR = Path(os.environ.get("GROUPED_DIR", DATA_DIR / "grouped"))
SAVE_PATH = Path(os.environ.get("SAVE_PATH", WEIGHTS_DIR / "mlp_head_triplet.pth"))
LOSS_CURVE_PATH = Path(os.environ.get("LOSS_CURVE_PATH", OUTPUT_DIR / "loss_curve_triplet.png"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
EPOCHS = int(os.environ.get("EPOCHS", "500"))
LR = float(os.environ.get("LR", "5e-4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
MLP_HIDDEN = int(os.environ.get("MLP_HIDDEN", "2048"))
MLP_HIDDEN2 = int(os.environ.get("MLP_HIDDEN2", "1024"))
MLP_OUT = int(os.environ.get("MLP_OUT", "512"))
IMG_SIZE = (224, 224)
VAL_FRACTION = float(os.environ.get("VAL_FRACTION", "0.2"))
TRIPLET_MARGIN = float(os.environ.get("TRIPLET_MARGIN", "0.5"))
LR_SCHEDULER = os.environ.get("LR_SCHEDULER", "cosine").lower()
USE_EARLY_STOPPING = os.environ.get("USE_EARLY_STOPPING", "0").lower() not in ("0", "false", "f")
EARLY_STOPPING_PATIENCE = int(os.environ.get("EARLY_STOPPING_PATIENCE", "10"))
USE_HARD_TRIPLET = os.environ.get("USE_HARD_TRIPLET", "false").lower() in ("true", "1", "yes")
NEG_WEIGHT = float(os.environ.get("NEG_WEIGHT", "2.0"))


class TripletDataset(Dataset):
    """Folder-based triplet dataset.

    Expected layout:
        data/grouped/class_001/*.jpg
        data/grouped/class_002/*.jpg
    """

    def __init__(self, root_dir, transform=None, split="train", val_fraction=0.2, seed=42):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.val_fraction = val_fraction
        random.seed(seed)

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.root_dir}")

        self.class_to_imgs = {}
        for class_dir in sorted(p for p in self.root_dir.iterdir() if p.is_dir()):
            imgs = sorted(
                p for p in class_dir.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            if len(imgs) < 2:
                continue

            random.shuffle(imgs)
            split_idx = max(1, int(len(imgs) * (1 - val_fraction)))
            selected = imgs[:split_idx] if split == "train" else imgs[split_idx:]
            if len(selected) >= 2:
                self.class_to_imgs[str(class_dir)] = [str(p) for p in selected]

        self.classes = list(self.class_to_imgs.keys())
        if not self.classes:
            raise ValueError(
                f"No valid classes found in {self.root_dir}. "
                "Each class folder needs at least two jpg/png images."
            )

    def __len__(self):
        return max(1, sum(len(v) for v in self.class_to_imgs.values()) * 4)

    def __getitem__(self, idx):
        anchor_cls = random.choice(self.classes)
        imgs_in_cls = self.class_to_imgs[anchor_cls]
        anchor_path, positive_path = random.sample(imgs_in_cls, 2)

        other_classes = [c for c in self.classes if c != anchor_cls]
        neg_cls = random.choice(other_classes) if other_classes else anchor_cls
        negative_candidates = [p for p in self.class_to_imgs[neg_cls] if p != anchor_path]
        negative_path = random.choice(negative_candidates)

        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")
        negative = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative


def pool_feature_for_mlp(feat):
    """Match the 128x7x7 flattened feature shape used by the saved MLP weights."""
    if feat.ndim != 4:
        raise ValueError(f"Expected 4D backbone output, got {tuple(feat.shape)}")
    return F.adaptive_max_pool2d(feat, (7, 7))


def hard_triplet_loss(anchor, positive, negative, margin=TRIPLET_MARGIN, neg_weight=2.0):
    pos_dist = torch.norm(anchor - positive, dim=1)
    neg_dist = torch.norm(anchor - negative, dim=1)
    return torch.relu(pos_dist - neg_weight * neg_dist + margin).mean()


def build_transform():
    return T.Compose(
        [
            T.Resize(IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_backbone():
    backbone = HybridCnnDinoNet(
        dino_ckpt_path=CKPT_PATH,
        dino_feature_layer=[23],
        superpoint_weight_path=SUPERPOINT_WEIGHT_PATH,
        load_ckpt=True,
    ).to(DEVICE)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone


def infer_feature_dim(backbone):
    with torch.no_grad():
        dummy = torch.zeros(1, 3, *IMG_SIZE).to(DEVICE)
        feat = pool_feature_for_mlp(backbone(dummy))
        return feat.shape[1] * feat.shape[2] * feat.shape[3]


def train():
    import matplotlib.pyplot as plt

    set_global_determinism(42)
    transform = build_transform()
    train_dataset = TripletDataset(GROUPED_DIR, transform=transform, split="train", val_fraction=VAL_FRACTION)
    val_dataset = TripletDataset(GROUPED_DIR, transform=transform, split="val", val_fraction=VAL_FRACTION)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train classes: {len(train_dataset.classes)}, samples: {len(train_dataset)}")
    print(f"Val classes: {len(val_dataset.classes)}, samples: {len(val_dataset)}")

    backbone = build_backbone()
    feat_dim = infer_feature_dim(backbone)
    print(f"Backbone feature dim: {feat_dim}")

    mlp_head = MLPHead(feat_dim, MLP_HIDDEN, MLP_HIDDEN2, MLP_OUT).to(DEVICE)
    optimizer = optim.Adam(mlp_head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2.0)

    if LR_SCHEDULER == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS))

    loss_list = []
    val_loss_list = []
    best_val_loss = float("inf")
    patience = 0

    for epoch in range(EPOCHS):
        mlp_head.train()
        total_loss = 0.0
        num_batches = 0

        for anchor, positive, negative in train_loader:
            anchor = anchor.to(DEVICE)
            positive = positive.to(DEVICE)
            negative = negative.to(DEVICE)

            with torch.no_grad():
                feat_anchor = pool_feature_for_mlp(backbone(anchor)).reshape(anchor.size(0), -1)
                feat_positive = pool_feature_for_mlp(backbone(positive)).reshape(positive.size(0), -1)
                feat_negative = pool_feature_for_mlp(backbone(negative)).reshape(negative.size(0), -1)

            proj_anchor = mlp_head(feat_anchor)
            proj_positive = mlp_head(feat_positive)
            proj_negative = mlp_head(feat_negative)

            loss = (
                hard_triplet_loss(proj_anchor, proj_positive, proj_negative, TRIPLET_MARGIN, NEG_WEIGHT)
                if USE_HARD_TRIPLET
                else criterion(proj_anchor, proj_positive, proj_negative)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        loss_list.append(avg_loss)

        mlp_head.eval()
        val_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor = anchor.to(DEVICE)
                positive = positive.to(DEVICE)
                negative = negative.to(DEVICE)

                feat_anchor = pool_feature_for_mlp(backbone(anchor)).reshape(anchor.size(0), -1)
                feat_positive = pool_feature_for_mlp(backbone(positive)).reshape(positive.size(0), -1)
                feat_negative = pool_feature_for_mlp(backbone(negative)).reshape(negative.size(0), -1)

                proj_anchor = mlp_head(feat_anchor)
                proj_positive = mlp_head(feat_positive)
                proj_negative = mlp_head(feat_negative)

                vloss = (
                    hard_triplet_loss(proj_anchor, proj_positive, proj_negative, TRIPLET_MARGIN, NEG_WEIGHT)
                    if USE_HARD_TRIPLET
                    else criterion(proj_anchor, proj_positive, proj_negative)
                )
                val_total += vloss.item()
                val_batches += 1

        avg_val_loss = val_total / max(1, val_batches)
        val_loss_list.append(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{EPOCHS} TrainLoss: {avg_loss:.4f} ValLoss: {avg_val_loss:.4f} LR: {current_lr:.6e}")

        if LR_SCHEDULER == "plateau":
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(mlp_head.state_dict(), SAVE_PATH)
            print(f"[Epoch {epoch + 1}] Saved best MLP weights: {SAVE_PATH}")
        else:
            patience += 1
            if USE_EARLY_STOPPING and patience >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping after {EARLY_STOPPING_PATIENCE} epochs without improvement.")
                break

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_list) + 1), loss_list, marker="o", label="train")
    plt.plot(range(1, len(val_loss_list) + 1), val_loss_list, marker="x", label="val")
    plt.xlabel("Epoch")
    plt.ylabel("TripletMarginLoss")
    plt.title("Training & Validation Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH, dpi=100)
    plt.close()
    print(f"Training complete. Loss curve saved to: {LOSS_CURVE_PATH}")


if __name__ == "__main__":
    train()
