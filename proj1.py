import os

# Define the path to the extracted dataset
data_dir = "A:/ai_project/traffic-detection-project"

# Check if the dataset directory exists
exists = os.path.exists(data_dir)
if exists:
    print(f"Dataset found at: {data_dir}")
else:
    print(f"Dataset not found at: {data_dir}. Please check the extraction process.")
