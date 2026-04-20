from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from DinoNet import DinoNet


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_SUPERPOINT_PATH = ROOT_DIR / "weights" / "superpoint_v1.pth"


def set_global_determinism(seed=42):
    """Set random seeds for reproducible feature extraction."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HybridCnnDinoNet(nn.Module):
    """CNN features guided by a DINOv2 spatial attention map."""

    def __init__(
        self,
        dino_ckpt_path,
        dino_feature_layer=None,
        superpoint_weight_path=DEFAULT_SUPERPOINT_PATH,
        load_ckpt=True,
        seed=42,
    ):
        set_global_determinism(seed)
        super().__init__()

        if dino_feature_layer is None:
            dino_feature_layer = [23]

        self.dino = DinoNet(
            cpt_path=str(dino_ckpt_path),
            feature_layer=dino_feature_layer,
            load_ckpt=load_ckpt,
        )
        self.cnn = self._build_superpoint_conv()

        if superpoint_weight_path:
            self._load_superpoint_weights(self.cnn, superpoint_weight_path)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            dino_feat = self.dino.extract_feature(dummy)
            dino_out_dim = dino_feat.shape[1]

        self.dino_out_dim = dino_out_dim
        self.dino_attn_proj = nn.Sequential(
            nn.Conv2d(dino_out_dim, 1, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _build_superpoint_conv():
        class SuperPointConv(nn.Module):
            def __init__(self):
                super().__init__()
                c1, c2, c3, c4 = 64, 64, 128, 128
                self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
                self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
                self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
                self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
                self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
                self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
                self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
                self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
                self.relu = nn.ReLU(inplace=True)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            def forward(self, x):
                if x.shape[1] == 3:
                    x = x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114
                x = self.relu(self.conv1a(x))
                x = self.relu(self.conv1b(x))
                x = self.pool(x)
                x = self.relu(self.conv2a(x))
                x = self.relu(self.conv2b(x))
                x = self.pool(x)
                x = self.relu(self.conv3a(x))
                x = self.relu(self.conv3b(x))
                x = self.pool(x)
                x = self.relu(self.conv4a(x))
                x = self.relu(self.conv4b(x))
                return x

        return SuperPointConv()

    @staticmethod
    def _load_superpoint_weights(model, weight_path):
        weight_path = Path(weight_path)
        if not weight_path.exists():
            raise FileNotFoundError(f"SuperPoint weights not found: {weight_path}")

        state_dict = torch.load(str(weight_path), map_location="cpu")
        allowed = ["conv1a", "conv1b", "conv2a", "conv2b", "conv3a", "conv3b", "conv4a", "conv4b"]
        filtered = {k: v for k, v in state_dict.items() if any(k.startswith(a) for a in allowed)}
        model.load_state_dict(filtered, strict=False)

    def extract_cnn_feature(self, x):
        return self.cnn(x)

    def extract_dino_attention(self, x, target_size):
        feats = self.dino.extract_feature(x)
        feat = torch.stack(feats, dim=0).mean(dim=0) if isinstance(feats, (tuple, list)) else feats

        if feat.shape[-2:] != target_size:
            feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
        return self.dino_attn_proj(feat)

    def forward(self, x, mask=None):
        feat_cnn = self.extract_cnn_feature(x)
        attn_map = self.extract_dino_attention(x, feat_cnn.shape[-2:])
        guided_feat = feat_cnn * attn_map

        if mask is not None:
            mask = mask.unsqueeze(1).float()
            mask = F.interpolate(mask, size=guided_feat.shape[-2:], mode="nearest")
            guided_feat = guided_feat * mask

        return guided_feat
