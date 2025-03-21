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
    # Define class mapping
    class_mapping = {
        'battery': 0,
        'biological': 1,
        'brown-glass': 2,
        'cardboard': 3,
        'clothes': 4,
        'green-glass': 5,
        'metal': 6,
        'paper': 7,
        'plastic': 8,
        'shoes': 9,
        'trash': 10,
        'white-glass': 11
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
                dst = os.path.join(f'data/images/{split}', img)
                shutil.copy2(src, dst)
                
                # Create YOLO format label (full image bounding box)
                label_path = os.path.join(f'data/labels/{split}', img.rsplit('.', 1)[0] + '.txt')
                with open(label_path, 'w') as f:
                    # Format: class x_center y_center width height
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
    
    # Create data.yaml file
    workspace_dir = os.path.abspath(os.path.dirname(__file__))
    yaml_content = f"""
train: {os.path.join(workspace_dir, 'data/images/train')}
val: {os.path.join(workspace_dir, 'data/images/val')}
test: {os.path.join(workspace_dir, 'data/images/test')}

nc: {len(class_mapping)}
names: {list(class_mapping.keys())}
    """
    
    with open('data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("Dataset preparation completed successfully")
    return True

def train_model():
    """Train the YOLOv8 model on the TrashNet dataset."""
    # Initialize model
    model = YOLO('yolov8n.pt')  # Load pretrained YOLOv8n model
    
    # Training parameters
    train_args = {
        'data': 'data.yaml',
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'workers': 8,
        'patience': 50,
        'project': 'runs/train',
        'name': 'trash_sorter',
        'exist_ok': True
    }
    
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
    print(f"Best model saved at: {results.best_model}")

if __name__ == "__main__":
    main()