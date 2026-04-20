from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import dino


class DinoNet(nn.Module):
    """DINOv2 Large feature extractor.

    This wrapper only keeps model loading and feature extraction logic.
    Visualization, heatmap generation, PCA, and file output utilities have been
    removed from this file.
    """

    def __init__(self, cpt_path=None, feature_layer=1, load_ckpt=True, device=None):
        super().__init__()
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.feature_layer = feature_layer

        self.model = dino.vit_large()
        if load_ckpt and cpt_path:
            state_dict = self._load_checkpoint(cpt_path)
            self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.h_down_rate = self.model.patch_embed.patch_size[0]
        self.w_down_rate = self.model.patch_embed.patch_size[1]

    @staticmethod
    def _load_checkpoint(cpt_path):
        cpt_path = Path(cpt_path)
        if not cpt_path.exists():
            raise FileNotFoundError(f"DINOv2 checkpoint not found: {cpt_path}")

        try:
            checkpoint = torch.load(str(cpt_path), map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(str(cpt_path), map_location="cpu")

        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                checkpoint = checkpoint["state_dict"]
            elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                checkpoint = checkpoint["model"]

        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unsupported DINOv2 checkpoint format: {cpt_path}")

        return {key.replace("module.", ""): value for key, value in checkpoint.items()}

    @staticmethod
    def _numpy_to_tensor(image):
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Unsupported numpy image shape: {image.shape}")

        return torch.from_numpy(image).float().permute(2, 0, 1)

    def extract_feature(self, image):
        """Extract DINOv2 patch features.

        Args:
            image: Tensor with shape (B, 3, H, W) or (3, H, W), or numpy image
                with shape (H, W, 3). Tensor inputs should already be normalized.

        Returns:
            Tensor with shape (B, C, H / 14, W / 14), or (C, H / 14, W / 14)
            when the input is a single image.
        """
        if isinstance(image, np.ndarray):
            image = self._numpy_to_tensor(image)
        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor or numpy.ndarray")

        image = image.to(self.device)
        was_single = False
        if image.ndim == 3:
            image = image.unsqueeze(0)
            was_single = True

        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected image shape (B, 3, H, W), got {tuple(image.shape)}")

        batch_size, _, height, width = image.shape
        if height % self.h_down_rate != 0 or width % self.w_down_rate != 0:
            raise ValueError(
                f"Image height and width must be multiples of "
                f"{self.h_down_rate}x{self.w_down_rate}, got {height}x{width}."
            )

        with torch.no_grad():
            output = self.model.get_intermediate_layers(image, n=self.feature_layer)[0]

        out_height = height // self.h_down_rate
        out_width = width // self.w_down_rate
        channels = output.shape[-1]
        features = output.reshape(batch_size, out_height, out_width, channels).permute(0, 3, 1, 2)
        features = 2 * features - 1

        if was_single:
            features = features.squeeze(0)
        return features

    def forward(self, image):
        return self.extract_feature(image)
