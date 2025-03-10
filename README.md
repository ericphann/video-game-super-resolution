# 🎮 Video Game Super-Resolution (VGSR) 🖥️
Project for DSBA 6165: Artificial Intelligence and Deep Learning

## Team

- Eric Phann
- Samantha Michael
- Kidus Kidane

## Overview

This project explores deep learning approaches to image super-resolution specifically for video game imagery. Rather than creating an end-user product, we aim to advance research in this field by comparing traditional convolutional and GAN-based architectures with newer transformer models. Inspired by technologies like NVIDIA's DLSS, our research investigates which techniques best preserve the unique visual characteristics of video games when upscaling from lower to higher resolutions. Our findings could inform future development of more accessible super-resolution solutions for game graphics.

## Problem Statement

Modern video games are increasingly demanding on hardware. Technologies like Deep Learning Super Sampling (DLSS) allow games to run at lower native resolutions for better performance, then upscale the imagery to higher resolutions using neural networks. This project explores open source approaches to this problem.

## Common Use Case

- Run games at lower resolution (better performance)
- Upscale to native resolution via super-resolution models
- Enable lower-end GPU users to enjoy graphically intensive games

## Approaches

We're exploring three primary neural network architectures:

1. **Convolutional Neural Networks (CNNs)**
   - Building upon pioneering work like SRCNN and VDSR
   - Focus on efficiency and performance

2. **Generative Adversarial Networks (GANs)**
   - Inspired by SRGAN and ESRGAN
   - Emphasis on perceptual quality and realistic textures

3. **Transformers**
   - Exploring newer architectures for super-resolution
   - Investigating attention mechanisms for detail preservation

## Datasets Explored

After thorough evaluation of multiple datasets (documented in our EDA folder), we selected the [**Super Resolution in Video Games Dataset (SRVG)**](https://www.kaggle.com/competitions/super-resolution-in-video-games/data) as our primary training data. This dataset consists of paired 270p and 1080p images from Unreal Engine projects, providing ideal content for training models specifically for game graphics upscaling.

Other datasets we explored include:

- [**Gameplay Images**](https://www.kaggle.com/datasets/aditmagotra/gameplay-images): Dataset containing 10 games with 1000 images per game

- [**Qualcomm Rasterized Images Dataset**](https://www.qualcomm.com/developer/software/qualcomm-rasterized-images-dataset): Parallel captures in different modalities and resolutions
   
Our dataset selection process and comparative analysis can be found in the `EDA/Data Collection & Analysis.pdf` file.

### Accessing Our Dataset
We've made our processed dataset available on Hugging Face for easy access and reproducibility. You can find it at: [ericphann/video-game-super-resolution](https://huggingface.co/datasets/ericphann/video-game-super-resolution/tree/main)

The dataset is organized in the following structure:
- `train-lr`: Low-resolution training images
- `train-hr`: High-resolution training images
- `validation-lr`: Low-resolution validation images
- `validation-hr`: High-resolution validation images
- `test-lr`: Low-resolution test images
- `test-hr`: High-resolution test images

#### Prerequisites
Install the required packages:
```
bash
pip install datasets huggingface-hub Pillow
```
To import the dataset in your code, use the following snippet:
```
from datasets import load_dataset
import os

# Load the dataset
dataset = load_dataset("ericphann/video-game-super-resolution")

# If you need to process the paired LR-HR images manually:
from PIL import Image
import glob

# Example: Loading training pairs
def load_image_pairs(base_path, split="train"):
    lr_path = os.path.join(base_path, f"{split}-lr")
    hr_path = os.path.join(base_path, f"{split}-hr")
    
    lr_files = sorted(glob.glob(os.path.join(lr_path, "*.png")))
    hr_files = sorted(glob.glob(os.path.join(hr_path, "*.png")))
    
    pairs = []
    for lr_file, hr_file in zip(lr_files, hr_files):
        lr_img = Image.open(lr_file)
        hr_img = Image.open(hr_file)
        pairs.append((lr_img, hr_img))
    
    return pairs

# Usage example
training_pairs = load_image_pairs("path/to/downloaded/dataset", "train")
validation_pairs = load_image_pairs("path/to/downloaded/dataset", "validation")
```

This dataset contains our preprocessed pairs of low-resolution and high-resolution game images, ready for training super-resolution models.

## Development Timeline

1. Review and submit EDA notebooks
2. Explore transformer-based approaches
3. Develop baseline models (CNN, GAN, Transformer)
4. Refine and optimize models
5. Final evaluation and comparison
6. Presentation and report submission

## Learning Outcomes

- Understanding traditional super-resolution approaches (CNNs, GANs)
- Experience with deep learning for processing and handling large image datasets
- Applications of super-resolution in the video game industry
- Deeper understanding of computer vision techniques

## Contribution Guidelines

We welcome contributions in the following areas:

- **Performance Improvements**:
  - Experimenting with deeper architectures
  - Optimizing for specific use cases
  - Adding multi-channel (RGB) support

- **Model Innovations**:
  - Novel loss functions
  - Hybrid architectures
  - Training optimizations

## References

- [SRCNN (2014)](https://arxiv.org/abs/1501.00092)
- [VDSR (2016)](https://arxiv.org/abs/1511.04587)
- [SRGAN (2017)](https://arxiv.org/abs/1609.04802)
- [ESRGAN (2018)](https://arxiv.org/abs/1809.00219)
- [Open Source SRCNN Implementation](https://github.com/yjn870/SRCNN-pytorch)
