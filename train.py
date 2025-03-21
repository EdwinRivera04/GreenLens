import os
import kagglehub
from ultralytics import YOLO
from pathlib import Path
import torch
import shutil
import sys
import random

def download_dataset():
    """Download the TrashNet dataset from Kaggle."""
    print("Checking for existing dataset...")
    
    # Check if data directory exists and is populated
    if os.path.exists('data/trashnet') and any(os.listdir('data/trashnet')):
        print("Dataset already exists in data/trashnet")
        return True
        
    print("Dataset not found. Starting download process...")
    try:
        # Check if Kaggle credentials exist
        kaggle_dir = os.path.expanduser('~/.kaggle')
        if not os.path.exists(kaggle_dir):
            print("Error: Kaggle credentials not found!")
            print("Please follow these steps:")
            print("1. Go to your Kaggle account settings")
            print("2. Click on 'Create New API Token'")
            print("3. Save the kaggle.json file to ~/.kaggle/kaggle.json")
            print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
            return False

        # Create necessary directories
        os.makedirs('data/images', exist_ok=True)
        os.makedirs('data/labels', exist_ok=True)
        os.makedirs('data/trashnet', exist_ok=True)

        # Download the dataset using kaggle-api instead of kagglehub
        print("Attempting to download dataset...")
        cmd = "kaggle datasets download -d mostafaabla/garbage-classification -p data/trashnet --unzip"
        os.system(cmd)
        
        # Verify the download
        if not os.path.exists('data/trashnet/garbage_classification'):
            print("Error: Dataset download failed or directory structure is incorrect")
            return False
            
        # Move files to correct locations
        source_dir = 'data/trashnet/garbage_classification'
        for category in os.listdir(source_dir):
            category_path = os.path.join(source_dir, category)
            if os.path.isdir(category_path):
                dest_path = os.path.join('data/trashnet', category.lower())
                print(f"Moving {category} to {dest_path}")
                shutil.move(category_path, dest_path)
                
        # Clean up
        if os.path.exists(source_dir):
            shutil.rmtree(source_dir)
            
        print("Dataset successfully downloaded and organized")
        return True
            
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Full error traceback:")
        traceback.print_exc()
        return False

def prepare_dataset():
    """Prepare the TrashNet dataset for YOLOv8 training."""
    # Define class mapping for the actual dataset
    class_mapping = {
        'cardboard': 0,
        'glass': 1,
        'metal': 2,
        'paper': 3,
        'plastic': 4,
        'trash': 5
    }
    
    # Create train/val/test splits
    splits = {
        'train': 0.7,
        'val': 0.2,
        'test': 0.1
    }
    
    # Create directories for splits
    for split in splits.keys():
        os.makedirs(f'data/images/{split}', exist_ok=True)
        os.makedirs(f'data/labels/{split}', exist_ok=True)
    
    # Process each category
    for category, class_id in class_mapping.items():
        category_dir = f'data/trashnet/{category}'
        if not os.path.exists(category_dir):
            print(f"Category directory not found: {category_dir}")
            return False
            
        # Get all images in category
        images = [f for f in os.listdir(category_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            print(f"No images found in {category_dir}")
            continue
            
        # Shuffle images
        random.shuffle(images)
        
        # Split images
        n_images = len(images)
        n_train = int(n_images * splits['train'])
        n_val = int(n_images * splits['val'])
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Move images and create labels
        for split, split_images in [('train', train_images), ('val', val_images), ('test', test_images)]:
            for img in split_images:
                # Copy image
                src = os.path.join(category_dir, img)
                dst = os.path.join(f'data/images/{split}', f"{category}_{img}")  # Add category prefix
                shutil.copy2(src, dst)
                
                # Create YOLO format label (full image bounding box)
                label_path = os.path.join(f'data/labels/{split}', f"{category}_{img.rsplit('.', 1)[0]}.txt")
                with open(label_path, 'w') as f:
                    # Format: class x_center y_center width height
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
    
    # Create data.yaml file
    workspace_dir = os.path.abspath(os.path.dirname(__file__))
    yaml_content = f"""path: {workspace_dir}  # dataset root dir
train: data/images/train  # train images
val: data/images/val  # val images
test: data/images/test  # test images

nc: {len(class_mapping)}  # number of classes
names: {list(class_mapping.keys())}  # class names
"""
    
    with open('data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("Dataset preparation completed successfully")
    return True

def train_model():
    """Train the YOLOv8 model on the TrashNet dataset."""
    # Initialize model
    model = YOLO('yolov8n.pt')  # Load pretrained YOLOv8n model
    
    # Device selection logic
    if torch.cuda.is_available():
        device = 'cuda'
        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}")
        print(f"CUDA Version: {torch.version.cuda}")
        # Balanced settings for better generalization
        batch_size = 16  # Small batch size for better generalization
        num_workers = 8
        mixed_precision = True
        image_size = 640  # Full resolution for better feature learning
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print("Using Apple Silicon GPU (MPS)")
        batch_size = 16
        num_workers = 4
        mixed_precision = True
        image_size = 640
    else:
        device = 'cpu'
        print("Using CPU")
        batch_size = 8
        num_workers = 4
        mixed_precision = False
        image_size = 640
    
    # Training parameters optimized for 91-95% accuracy
    train_args = {
        'data': 'data.yaml',
        'epochs': 40,  # More epochs for thorough learning
        'imgsz': image_size,
        'batch': batch_size,
        'device': device,
        'workers': num_workers,
        'patience': 10,  # Early stopping to prevent overfitting
        'project': 'runs/train',
        'name': 'trash_sorter_balanced',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',  # AdamW with weight decay for regularization
        'amp': mixed_precision,
        'lr0': 0.001,  # Initial learning rate
        'lrf': 0.01,   # Final learning rate ratio
        'momentum': 0.937,
        'weight_decay': 0.005,  # Moderate weight decay
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.3,  # Reduced classification loss weight
        'dfl': 1.5,
        'plots': True,
        'save': True,
        'cache': True,
        'rect': False,  # Disabled for better generalization
        'cos_lr': True,  # Cosine learning rate schedule
        'close_mosaic': 10,
        'dropout': 0.15,  # Moderate dropout
        'label_smoothing': 0.05,  # Light label smoothing
        'augment': True,
        'mixup': 0.15,  # Moderate mixup
        'copy_paste': 0.15,  # Moderate copy-paste
        'degrees': 15.0,  # Rotation augmentation
        'translate': 0.2,  # Translation augmentation
        'scale': 0.5,  # Scale augmentation
        'shear': 5.0,  # Shear augmentation
        'perspective': 0.001,  # Light perspective augmentation
        'flipud': 0.3,  # Vertical flip
        'fliplr': 0.5,  # Horizontal flip
        'mosaic': 0.8,  # Reduced mosaic for better stability
        'hsv_h': 0.015,  # HSV augmentation
        'hsv_s': 0.5,
        'hsv_v': 0.3,
        'overlap_mask': True,  # Better mask handling
        'mask_ratio': 4,
        'val_check_interval': 0.5,  # Validate more frequently
        'save_period': -1,  # Save only best and last
    }
    
    print(f"\nBalanced Training Configuration:")
    print(f"Target Accuracy: 91-95%")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {image_size}")
    print(f"Epochs: {train_args['epochs']}")
    print(f"Learning rate: {train_args['lr0']}")
    print(f"Weight decay: {train_args['weight_decay']}")
    print(f"Dropout: {train_args['dropout']}")
    print(f"Label smoothing: {train_args['label_smoothing']}")
    print("Augmentations: rotation, translation, scale, shear, mixup")
    print("Early stopping patience: 10 epochs")
    print("Estimated time: 35-45 minutes on RTX 4070 Ti\n")
    
    # Start training
    results = model.train(**train_args)
    
    return results

def main():
    print("Starting Trash Sorter training process...")
    
    # Download and prepare dataset
    if not download_dataset():
        print("Dataset download failed.")
        return
        
    if not prepare_dataset():
        print("Dataset preparation failed. Please check the dataset structure.")
        return
    
    # Train model
    print("Starting model training...")
    results = train_model()
    
    print("Training completed!")
    print("Best model saved at: runs/train/trash_sorter_v2/weights/best.pt")
    print("Last model saved at: runs/train/trash_sorter_v2/weights/last.pt")
    print(f"Final mAP50: {results.box.map50:.3f}")
    print(f"Final mAP50-95: {results.box.map:.3f}")

if __name__ == "__main__":
    main()