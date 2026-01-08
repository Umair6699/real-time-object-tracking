# Real-Time Object Tracking and Trajectory Mapping  
### Using Sparse Optical Flow and Kalman Filtering

## 📌 Project Overview

This project implements a **real-time object tracking and trajectory mapping system** using **classical computer vision techniques**. The system tracks a user-selected object in video sequences and estimates a smooth motion trajectory without relying on deep learning models.

The approach integrates:
- **Sparse Optical Flow (Lucas–Kanade)**
- **RANSAC-based outlier rejection**
- **Kalman Filter–based motion smoothing**

The project focuses on **accuracy, efficiency, and interpretability**, making it suitable for real-time applications and resource-constrained systems.

---

## 🎯 Objectives

- Track moving objects in real time  
- Reduce noise in motion estimation  
- Handle outliers and partial occlusions  
- Avoid heavy deep learning models  
- Demonstrate classical computer vision pipelines  

---

## ✨ Key Features

- Manual **ROI-based object initialization**
- Shi–Tomasi corner detection for feature extraction
- Pyramidal Lucas–Kanade sparse optical flow
- RANSAC-based outlier rejection
- Kalman filter for trajectory smoothing
- Real-time visualization of:
  - Feature points
  - Object centroid
  - Motion trajectory
- Lightweight and training-free implementation

---

## 🧠 System Pipeline

1. **ROI Selection**  
   User pauses the video and selects the target object.

2. **Feature Detection**  
   Shi–Tomasi corner detection extracts salient points.

3. **Sparse Optical Flow Tracking**  
   Lucas–Kanade method tracks features across frames.

4. **Outlier Rejection (RANSAC)**  
   Removes inconsistent feature matches.

5. **Kalman Filtering**  
   Smooths noisy centroid estimates using a constant velocity model.

6. **Trajectory Mapping**  
   Displays object motion over time.

---

## 🧮 Kalman Filter Model

The motion model uses a **constant velocity state vector**:


This model improves robustness against noise and short-term tracking failures.

---

## 🛠️ Technologies Used

- Python  
- OpenCV  
- NumPy  
- Classical Computer Vision Algorithms  
- Kalman Filtering  

---

---

## 📑 Documentation

### 📄 Project Report
- **CV_Project_Report_Final.pdf**
- Contains:
  - Optical flow theory
  - Kalman filter formulation
  - RANSAC-based outlier rejection
  - Experimental discussion
  - References

### 📊 Presentation Slides
- **Real-Time Object Tracking and Trajectory Mapping using Sparse Optical Flow**
- Used for academic presentation and evaluation

---

## ⚠️ Limitations

- Requires manual ROI selection  
- Performance degrades under long-term occlusion  
- Designed for single-object tracking  
- Very fast object motion may reduce tracking accuracy  

---

## 🚀 Future Improvements

- Automatic object detection  
- Multi-object tracking  
- Integration with deep learning detectors  
- Hardware acceleration (FPGA / Edge AI)  
- Real-time deployment on embedded platforms  

---

## 👤 Authors

**Muhammad Umair Ajmal**  
**Zeeshan Haider**  
**Muhammad Huzaifa**

---

## 📌 Project Status

✔ Complete implementation  
✔ Verified on real video sequences  
✔ Documentation and presentation included  

