# Chip Layout Similarity

This repository provides code for chip layout image similarity matching based on a Hybrid CNN + DINOv2 Large backbone and an MLP projection head.

The method uses SuperPoint-style CNN features for local structure extraction, DINOv2 Large features for spatial attention guidance, and a trained MLP head to project image features into a similarity embedding space.

## Release Assets

We provide the required dataset and model weights through GitHub Releases:

```text
https://github.com/Myy666666/Chip-layout/releases
```

The Release assets include:

- A partial open-source chip layout dataset.
- DINOv2 Large / ViT-L/14 pretrained weights.
- SuperPoint pretrained weights.
- Our trained MLP projection-head weights.

These files are not stored directly in the Git repository because the dataset and model checkpoints are large. Please download them from the Release page before training or prediction.

## Expected File Structure

After cloning this repository and downloading the Release assets, place the files like this:

```text
Chip-layout/
  data/
    grouped/
      class_001/
        image_a.jpg
        image_b.jpg
      class_002/
        image_c.jpg
        image_d.jpg
  weights/
    dinov2_vitl14_pretrain.pth
    superpoint_v1.pth
    mlp_head_triplet-0410.pth
```

The names above are the default paths used by the scripts. If your files are stored elsewhere, you can set environment variables instead.

## Repository Contents

- `hybrid_cnn_dino.py`: Hybrid CNN + DINOv2 attention-guided backbone.
- `DinoNet.py`, `dino.py`, `dino_utils.py`: local DINOv2 ViT implementation.
- `mlp_head.py`: MLP projection head.
- `train_mlp_cosine.py`: triplet-loss training script.
- `predict_mlp_cosine.py`: pairwise chip layout similarity prediction script.
- `requirements.txt`: Python dependencies.

## Installation

Install the Python dependencies:

```cmd
pip install -r requirements.txt
```

If you use GPU acceleration, install the PyTorch version that matches your CUDA environment.

## Dataset Format

Training expects grouped class folders. Images in the same folder are treated as positive samples, and images from different folders are treated as negative samples.

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

Each class folder should contain at least two images.

To use a custom dataset path:

```cmd
set GROUPED_DIR=D:\path\to\grouped_dataset
```

## Training

Before training, download the DINOv2 Large weights and SuperPoint weights from Releases and place them in:

```text
weights/dinov2_vitl14_pretrain.pth
weights/superpoint_v1.pth
```

Run training:

```cmd
python train_mlp_cosine.py
```

Common options:

```cmd
set GROUPED_DIR=D:\path\to\grouped_dataset
set EPOCHS=100
set BATCH_SIZE=16
python train_mlp_cosine.py
```

The newly trained MLP weights are saved to:

```text
weights/mlp_head_triplet.pth
```

## Prediction

Before prediction, download all three weight files from Releases:

```text
weights/dinov2_vitl14_pretrain.pth
weights/superpoint_v1.pth
weights/mlp_head_triplet-0410.pth
```

Compare two chip layout images:

```cmd
python predict_mlp_cosine.py examples\image1.jpg examples\image2.jpg
```

Optional arguments:

```cmd
python predict_mlp_cosine.py image1.jpg image2.jpg --rotate-deg 90 --threshold 0.7 --show
```

If your DINOv2 checkpoint or MLP checkpoint is stored in another location, use:

```cmd
set DINO_CKPT_PATH=D:\path\to\dinov2_vitl14_pretrain.pth
set MLP_WEIGHT_PATH=D:\path\to\mlp_head_triplet-0410.pth
```

## Notes

- Large datasets and model checkpoints are distributed through GitHub Releases.
- The Git repository contains the source code and lightweight project files.
- The Release page contains the partial chip layout dataset and the three required weights: DINOv2 Large, SuperPoint, and the trained MLP head.
