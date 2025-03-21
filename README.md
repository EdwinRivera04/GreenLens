# Trash Sorting with YOLOv8

An automated trash sorting system using YOLOv8 for waste classification. This project uses deep learning to classify different types of waste materials, helping to improve recycling efficiency.

## Hardware Optimizations

The training script is optimized for different hardware configurations:

### Apple Silicon (M-series)
- Utilizes Metal Performance Shaders (MPS) for GPU acceleration
- Optimized for unified memory architecture
- For M4 Pro with 24GB RAM:
  - Batch size: 24
  - Worker threads: 6
  - Mixed precision training enabled
  - Image caching enabled
  - Expected training time: 2-3 hours

### NVIDIA GPUs
- CUDA acceleration support
- For RTX 4070 Ti (12GB VRAM):
  - Batch size: 32
  - Worker threads: 8
  - Mixed precision training enabled
  - Expected training time: 1-2 hours

### CPU
- Fallback option for systems without GPU
- Optimized parameters:
  - Batch size: 8
  - Worker threads: 4
  - Expected training time: 15-20 hours

## Dataset

Uses the Garbage Classification dataset from Kaggle, which includes:
- 12 waste categories
- ~15,000 images
- Categories: battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass

## Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:
- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- Kaggle API
- OpenCV

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Kaggle credentials:
   - Go to your Kaggle account settings
   - Click on 'Create New API Token'
   - Save the kaggle.json file to ~/.kaggle/kaggle.json
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## Usage

Run the training script:
```bash
python train.py
```

The script will:
1. Download and prepare the dataset
2. Automatically detect your hardware and optimize training parameters
3. Train the YOLOv8 model
4. Save the best model and training results

## Project Structure

```
├── data/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── trashnet/
├── runs/
│   └── train/
├── train.py
├── requirements.txt
└── README.md
```

## Training Configuration

- Model: YOLOv8n (nano)
- Image size: 640x640
- Epochs: 50
- Optimizer: Adam
- Learning rate: 0.001
- Data augmentation:
  - Horizontal flip
  - Translation
  - Scale
  - Mosaic

## Results

Training results and model checkpoints are saved in the `runs/train/trash_sorter` directory, including:
- Best model weights
- Training metrics
- Confusion matrix
- PR curves
- Training plots

## Hardware Requirements

Minimum:
- 8GB RAM
- 10GB storage space
- CPU with 4+ cores

Recommended:
- Apple M-series chip with 16GB+ RAM or NVIDIA GPU with 8GB+ VRAM
- 20GB+ storage space
- SSD for faster data loading

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: [Garbage Classification on Kaggle](https://www.kaggle.com/mostafaabla/garbage-classification)
- YOLOv8: [Ultralytics](https://github.com/ultralytics/ultralytics) 