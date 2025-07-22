import cv2
import os
import pandas as pd
from ultralytics import YOLO

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# Define dataset path
data_dir = "A:/ai_project/traffic-detection-project/train/images"
image_paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.jpg') or file.endswith('.png')]

# Function to estimate traffic density using YOLOv8
def estimate_traffic_density(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return None

    # Run YOLOv8 on the image
    results = yolo_model(image_path)
    vehicle_count = sum(1 for obj in results[0].boxes.cls if int(obj) in [2, 3, 5, 7])  # Car, Motorbike, Bus, Truck
    
    # Calculate traffic density
    traffic_density = vehicle_count / (img.shape[0] * img.shape[1])
    
    return vehicle_count, traffic_density

# Store densities in a DataFrame
data = []
for path in image_paths:
    vehicle_count, density = estimate_traffic_density(path)
    if density is not None:
        data.append([path, vehicle_count, density])

# Save traffic densities to CSV
df = pd.DataFrame(data, columns=["Image", "Vehicle_Count", "Traffic_Density"])
df.to_csv("traffic_densities.csv", index=False)

print("Traffic densities saved to traffic_densities.csv")