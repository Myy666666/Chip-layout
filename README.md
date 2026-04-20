# Chip Layout Similarity

This repository contains the implementation of a chip layout image similarity method based on a Hybrid CNN + DINOv2 Large backbone and an MLP projection head.

The model extracts local CNN features, uses DINOv2 features to generate spatial attention, and compares two chip layout images by cosine similarity in the learned embedding space.

## Open-Source Dataset and Weights

We have open-sourced part of the chip layout dataset and the required pretrained weights in the GitHub Releases of this repository:

```text
https://github.com/Myy666666/Chip-layout/releases
```

The Release assets include:

- A partial chip layout dataset for training and testing.
- DINOv2 Large / ViT-L/14 pretrained weights.
- SuperPoint pretrained weights.

After downloading the Release assets, place them in the following structure:

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

`dinov2_vitl14_pretrain.pth` is about 1.2 GB, so it is not stored directly in the Git repository. Please download it from Releases before running training or prediction.

## Repository Contents

- `hybrid_cnn_dino.py`: Hybrid CNN + DINOv2 attention-guided backbone.
- `DinoNet.py`, `dino.py`, `dino_utils.py`: local DINOv2 ViT implementation.
- `mlp_head.py`: MLP projection head.
- `train_mlp_cosine.py`: triplet-loss training script.
- `predict_mlp_cosine.py`: pairwise chip layout similarity prediction script.
- `weights/mlp_head_triplet-0410.pth`: trained MLP projection-head weights.
- `requirements.txt`: Python dependencies.

## Installation

Create a Python environment, then install dependencies:

```cmd
pip install -r requirements.txt
```

If PyTorch needs CUDA support, install the PyTorch version that matches your CUDA environment from the official PyTorch instructions.

## Dataset Format

Training uses grouped class folders. Images in the same folder are treated as positive samples, while images from different folders are treated as negative samples.

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

You can also use another dataset path by setting `GROUPED_DIR`:

```cmd
set GROUPED_DIR=D:\path\to\grouped_dataset
```

## Training

Make sure the DINOv2 Large weights and SuperPoint weights are available:

```text
weights/dinov2_vitl14_pretrain.pth
weights/superpoint_v1.pth
```

Then run:

```cmd
python train_mlp_cosine.py
```

Common training overrides:

```cmd
set GROUPED_DIR=D:\path\to\grouped_dataset
set EPOCHS=100
set BATCH_SIZE=16
python train_mlp_cosine.py
```

The trained MLP weights are saved to:

```text
weights/mlp_head_triplet.pth
```

## Prediction

Compare two chip layout images:

```cmd
python predict_mlp_cosine.py examples\image1.jpg examples\image2.jpg
```

Optional arguments:

```cmd
python predict_mlp_cosine.py image1.jpg image2.jpg --rotate-deg 90 --threshold 0.7 --show
```

If the DINOv2 checkpoint is stored somewhere else, set:

```cmd
set DINO_CKPT_PATH=D:\path\to\dinov2_vitl14_pretrain.pth
```

## Notes

- Large files such as the DINOv2 Large checkpoint and full datasets should be distributed through GitHub Releases, Git LFS, or external dataset hosting.
- The repository keeps the code and lightweight configuration in Git, while large assets are referenced from Releases.
- The included `mlp_head_triplet-0410.pth` can be used directly for prediction after the DINOv2 and SuperPoint weights are prepared.
