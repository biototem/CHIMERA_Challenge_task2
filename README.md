# CHIMERA Challenge Task 2: Attention-based MIL for Whole Slide Image Classification

## Overview

This repository contains the implementation for the CHIMERA Challenge Task 2, focusing on Multiple Instance Learning (MIL) with attention mechanisms for whole slide image (WSI) classification. The framework performs feature extraction from WSIs, trains attention-based MIL models using 5-fold cross-validation, and generates predictions for medical image analysis.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Usage](#usage)
- [Configuration](#configuration)
- [Docker Deployment](#docker-deployment)
- [Model Components](#model-components)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)


## Installation

### 1. Clone the Repository

```

git clone https://github.com/biototem/CHIMERA_Challenge_task2.git
cd CHIMERA_Challenge_task2

```

### 2. Create Virtual Environment

```

conda create -n py311 python=3.11
activate py311
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r inference/requirements.txt

```

## Project Structure

```
CHIMERA_Challenge_task2/
│
├── get_feat/                   # Feature extraction module
│   ├── main_feature.py         # Main feature extraction script
│   ├── read_get_index.py       # Index reading utilities
│   ├── tiffslide_utils.py      # TIFF slide handling utilities
│   └── tile_enhancement.py     # Tile augmentation functions
│
├── architecture/              # Model architectures
│   ├── transformer.py         # Transformer-based models
│   ├── .....                  # 
│
├── modules/                   # Reusable model components
│   ├── ...         
│
├── inference/                 # Inference pipeline
│   ├── inference.py           # Main inference script
│   ├── Dockerfile             # Docker container configuration
│   ├── requirements.txt       # Inference dependencies
│   └── architecture/          # Inference model architectures
│
├── t1_cfg.py                   # Configuration file
├── main_train_acmil.py         # Main training script
├── t1_feat_dataset.py          # Dataset loader for features
└── data_linc_8-8.pkl           # clinical features
```

## Workflow

The complete pipeline consists of four main stages:

### Stage 1: Feature Extraction

Extract features from whole slide images using pre-trained models ,Before performing feature extraction, please visit the H0-mini team to obtain model resources:
https://huggingface.co/bioptimus/H0-mini
```
python get_feat/main_feature.py
```

This script:
- Saves features and spatial coordinates for each slide
- Besides clinical features:Please refer to the read_data_json function in inference/reference.py
- already extracted the clinical features from the Chimera dataset and saved them in data_inc_8-8.pkl

### Stage 2: Configuration Setup

Configure training parameters in `t1_cfg.py`:

```
# Key configuration parameters
xxxxxxx111 = '4'        # Fold number (1-5)
feat_dim = 810          # Feature dimension
n_cls = 3               # Number of classes
train_number            #How many iterations per round
epoch = 400
# Data paths
feat_dir = '/path/to/features/'
feat_位置信息_dir = '/path/to/coordinates/'
```

### Stage 3: Model Training

Train the attention-based MIL model with 5-fold cross-validation:

```
python main_train_acmil.py
```

Training features:
- 5-fold cross-validation strategy
- Attention-based instance aggregation
- Loss calculation with weighted sampling
- Model checkpointing for best validation performance
- Metric tracking (AUC, F1-score, accuracy)

### Stage 4: Inference

Generate predictions on test data:

```
Please Refer to inference/inference.py
```

## Docker Deployment
### Building the Docker Image
- cd ./inference
- Place the trained Feature extraction model and MIL model in inference/resources/
- sudo docker build --platform linux/amd64 -t brs1:1.0 .



## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Citation
If you use this code in your research, please cite:

```bibtex
@article{chimera2025}
```
