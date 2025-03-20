import cv2
import numpy as np
from ultralytics import YOLO
import time

class TrashSorter:
    def __init__(self, model_path='models/best.pt'):
        """Initialize the trash sorter with a trained YOLOv8 model."""
        self.model = YOLO(model_path)
        self.categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        self.colors = {
            'cardboard': (255, 0, 0),    # Blue
            'glass': (0, 255, 0),        # Green
            'metal': (0, 0, 255),        # Red
            'paper': (255, 255, 0),      # Yellow
            'plastic': (255, 0, 255),    # Magenta
            'trash': (128, 128, 128)     # Gray
        }

    def process_frame(self, frame):
        """Process a single frame and return the annotated frame."""
        # Run YOLOv8 inference
        results = self.model(frame, conf=0.25)
        
        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get confidence and class
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                category = self.categories[cls]
                
                # Draw bounding box
                color = self.colors[category]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label = f'{category} {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame

def main():
    # Initialize the trash sorter
    sorter = TrashSorter()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set frame dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Trash Sorter is running. Press 'q' to quit.")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Process frame
        processed_frame = sorter.process_frame(frame)
        
        # Display frame
        cv2.imshow('Trash Sorter', processed_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 