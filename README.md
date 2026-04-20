# Chip Layout Similarity

This is a cleaned GitHub-ready copy of the Hybrid CNN + DINOv2 + MLP code for chip layout image similarity.

## Included Files

- `hybrid_cnn_dino.py`: Hybrid CNN + DINOv2 backbone.
- `DinoNet.py`, `dino.py`, `dino_utils.py`: local DINOv2 ViT implementation.
- `mlp_head.py`: MLP projection head.
- `train_mlp_cosine.py`: triplet-loss training script.
- `predict_mlp_cosine.py`: pairwise prediction script.
- `weights/superpoint_v1.pth`: SuperPoint CNN initialization weights.
- `weights/mlp_head_triplet-0410.pth`: trained MLP weights.

## Not Included

`weights/dinov2_vitl14_pretrain.pth` is about 1.2 GB, so it is not suitable for a normal GitHub push. Put that file here after cloning:

```text
weights/dinov2_vitl14_pretrain.pth
```

You can also point to another location with:

```cmd
set DINO_CKPT_PATH=D:\path\to\dinov2_vitl14_pretrain.pth
```

## Install

```cmd
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version from the official PyTorch instructions if needed.

## Dataset Layout

Training expects grouped class folders:

```text
data/
  grouped/
    class_001/
      image_a.jpg
      image_b.jpg
    class_002/
      image_c.jpg
      image_d.jpg
```

Each class folder needs at least two images.

## Train

```cmd
python train_mlp_cosine.py
```

Useful environment overrides:

```cmd
set GROUPED_DIR=D:\path\to\grouped_dataset
set EPOCHS=100
set BATCH_SIZE=16
python train_mlp_cosine.py
```

The best MLP weights are saved to:

```text
weights/mlp_head_triplet.pth
```

## Predict

```cmd
python predict_mlp_cosine.py examples\image1.jpg examples\image2.jpg
```

Optional arguments:

```cmd
python predict_mlp_cosine.py image1.jpg image2.jpg --rotate-deg 90 --threshold 0.7 --show
```

## GitHub Upload

This folder is intended to be uploaded as a separate repository. The 1.2 GB DINOv2 checkpoint is ignored by `.gitignore`; upload it elsewhere or use Git LFS if you really need it in GitHub.

