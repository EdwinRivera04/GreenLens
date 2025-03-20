import os
import kagglehub
from ultralytics import YOLO
from pathlib import Path
import torch
import shutil
import sys

def download_dataset():
    """Download the TrashNet dataset from Kaggle."""
    print("Downloading TrashNet dataset from Kaggle...")
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

        # Download the dataset
        print("Attempting to download dataset...")
        path = kagglehub.dataset_download("feyzazkefe/trashnet")
        print(f"Dataset downloaded to: {path}")
        
        if not path:
            print("Error: No path returned from kagglehub.dataset_download")
            return False
            
        # Create necessary directories
        os.makedirs('data/images', exist_ok=True)
        os.makedirs('data/labels', exist_ok=True)
        
        # Move the dataset to the correct location
        dataset_path = path[0]  # kagglehub returns a list of paths
        print(f"Processing dataset from: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset path not found: {dataset_path}")
            return False
            
        # Move the dataset to our data directory
        for item in os.listdir(dataset_path):
            s = os.path.join(dataset_path, item)
            d = os.path.join('data/trashnet', item)
            print(f"Moving {item} to {d}")
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        print("Dataset successfully moved to data/trashnet directory")
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
        'cardboard': 0,
        'glass': 1,
        'metal': 2,
        'paper': 3,
        'plastic': 4,
        'trash': 5
    }
    
    # Process each category
    for category in class_mapping.keys():
        category_dir = f'data/trashnet/{category}'
        if not os.path.exists(category_dir):
            print(f"Category directory not found: {category_dir}")
            return False
    
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