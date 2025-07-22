import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt
from ultralytics import YOLO

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")  # Use custom model if available

# Load trained ML model and scaler
ml_model = joblib.load("traffic_signal_model.pkl")
scaler = joblib.load("scaler.pkl")

# Vehicle class IDs (COCO dataset)
vehicle_classes = [2, 3, 5, 7]

# Parameters
image_paths = [
    r"A:\ai_project\traffic-detection-project\test\images\screenshot_26533_jpg.rf.4f20415f4d181828961118dda8c82a1b.jpg",
    r"A:\ai_project\traffic-detection-project\test\images\duque_de_caxias-2455_png_jpg.rf.9c781508aa471f79d9571df5df404b19.jpg",
    r"A:\ai_project\traffic-detection-project\test\images\ant_sales-10505_png_jpg.rf.fdecaeab0dfbdb686fd11aefc7ab74e7.jpg",
    r"A:\ai_project\traffic-detection-project\test\images\duque_de_caxias-1755_png_jpg.rf.4baca34696f1d0904200d0c58e2344af.jpg",
    r"A:\ai_project\traffic-detection-project\test\images\aguanambi-3125_png_jpg.rf.c928762f63b976739588cd27f57fcede.jpg",
    r"A:\ai_project\traffic-detection-project\test\images\ant_sales-10580_png_jpg.rf.bc97a488d81f01f82e0d25b6cf1dc930.jpg",
    r"A:\ai_project\traffic-detection-project\test\images\duque_de_caxias-405_png_jpg.rf.cdb0860def4d0b3a15ecc51682271f22.jpg",
    r"A:\ai_project\traffic-detection-project\test\images\duque_de_caxias-995_png_jpg.rf.8eb6ae806d383a68998e792fa8941d34.jpg",
    r"A:\ai_project\traffic-detection-project\test\images\duque_de_caxias-1880_png_jpg.rf.f341c902790abfe90d3173e36c52408c.jpg"
]

fixed_time = 45  # seconds
vehicle_flow_rate = 1 / 2  # vehicles per second (1 vehicle every 2 seconds)

# Lists to store results
vehicle_counts_all = []
predicted_times_all = []
fixed_efficiency_all = []
dynamic_efficiency_all = []
vehicles_cleared_fixed_all = []
vehicles_cleared_dynamic_all = []

# --- IMAGE PROCESSING & ANALYSIS ---
plt.figure(figsize=(18, 10))
cols = ceil(sqrt(len(image_paths)))
rows = ceil(len(image_paths) / cols)

for idx, path in enumerate(image_paths):
    # YOLO detection
    results = yolo_model(path)[0]
    detected_classes = results.boxes.cls.tolist()
    vehicle_count = sum(1 for cls_id in detected_classes if int(cls_id) in vehicle_classes)

    # ML prediction
    scaled_input = scaler.transform([[vehicle_count]])
    predicted_time = int(ml_model.predict(scaled_input)[0])

    # Simulate real vehicle clearance
    vehicles_cleared_fixed = min(vehicle_count, int(fixed_time * vehicle_flow_rate))
    vehicles_cleared_dynamic = min(vehicle_count, int(predicted_time * vehicle_flow_rate))

    efficiency_fixed = vehicles_cleared_fixed / fixed_time
    efficiency_dynamic = vehicles_cleared_dynamic / predicted_time

    # Save stats
    vehicle_counts_all.append(vehicle_count)
    predicted_times_all.append(predicted_time)
    vehicles_cleared_fixed_all.append(vehicles_cleared_fixed)
    vehicles_cleared_dynamic_all.append(vehicles_cleared_dynamic)
    fixed_efficiency_all.append(efficiency_fixed)
    dynamic_efficiency_all.append(efficiency_dynamic)

    # Plot image with info
    image = results.plot()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.subplot(rows, cols, idx + 1)
    plt.imshow(image)
    plt.title(f"Signal {idx+1}\n Vehicles Detected: {vehicle_count} | Signal Timer: {predicted_time}s", fontsize=10)
    plt.axis('off')

plt.suptitle("Dynamic Traffic Signal Timing using YOLO + ML", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.savefig("dynamic_signal_detection_grid.png", dpi=300)
plt.show()

# --- EFFICIENCY COMPARISON GRAPH ---
labels = [f"Signal {i+1}" for i in range(len(image_paths))]
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, fixed_efficiency_all, width, label='Fixed Timer (45s)', color='gray')
plt.bar(x + width/2, dynamic_efficiency_all, width, label='Dynamic Timer (ML Model)', color='green')

plt.ylabel('Vehicles per Second (Efficiency)', fontsize=12)
plt.title('Efficiency Comparison: Fixed vs Dynamic Traffic Timing', fontsize=14)
plt.xticks(x, labels)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("efficiency_comparison_fullset.png", dpi=300)
plt.show()

# --- CONSOLE SUMMARY ---
print("\n--- Traffic Signal Efficiency Summary ---")
for i in range(len(image_paths)):
    print(f"Signal {i+1}:")
    print(f"  Vehicles Detected          : {vehicle_counts_all[i]}")
    print(f"  Fixed Time (45s)           : Vehicles Cleared = {vehicles_cleared_fixed_all[i]} → Efficiency = {fixed_efficiency_all[i]:.2f} veh/sec")
    print(f"  Dynamic Time (Predicted={predicted_times_all[i]}s): Vehicles Cleared = {vehicles_cleared_dynamic_all[i]} → Efficiency = {dynamic_efficiency_all[i]:.2f} veh/sec\n")
