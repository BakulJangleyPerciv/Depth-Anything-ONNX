import onnxruntime as ort
import numpy as np
import cv2

# 1. Load Session
session = ort.InferenceSession("/home/bakul/repos/ORB-SLAM3-ROS2-Docker/egotrack_relay/metric_depth_vitb.onnx")

# 2. Load and Preprocess a REAL image
# 2. Load and Preprocess a REAL image
raw_img = cv2.imread("/home/bakul/repos/ORB-SLAM3-ROS2-Docker/thirdparty/Depth-Anything-ONNX/test.png") 
img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (518, 518))

# Explicitly use float32 for everything to satisfy ONNX
img = img.astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

img = (img - mean) / std
img = img.transpose(2, 0, 1)[np.newaxis, ...]
img = np.ascontiguousarray(img, dtype=np.float32) # Force float32 and memory layout

# 3. Inference
output = session.run(None, {'image': img})[0]
output = np.squeeze(output)

# 4. Check Variance (If variance is near 0, the model is a dud)
print(f"Max: {np.max(output)}, Min: {np.min(output)}, StdDev: {np.std(output)}")

# 5. Save a heatmap
depth_norm = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
heatmap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
cv2.imwrite("host_verification.png", heatmap)