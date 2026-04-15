🛰️ SVAMITVA Feature Extraction Pipeline

MoPR x IIT Tirupati – 2026

A full-scale deep learning pipeline for geospatial feature extraction from high-resolution satellite imagery using semantic segmentation and ensemble learning.

🚀 Overview

This project implements an end-to-end pipeline for extracting land features such as:

Buildings (RCC, Tiled, Tin, Thatched)
Roads
Water bodies
Transformers, Tanks, Wells

It combines:

Data preprocessing (GeoTIFF + shapefiles)
Patch-based training
Multiple segmentation models
Ensemble learning
Geospatial output generation (GeoPackage)

🧠 Models Used
The system uses a hybrid ensemble of 3 models:

SegFormer (B5) – Transformer-based segmentation
Mask2Former – Universal segmentation model
DuSA U-Net++ – Custom CNN with attention + ASPP

Final predictions are generated using a weighted ensemble.

⚙️ Features
✅ Data Preparation
Converts .ecw → .tif
Rasterizes shapefiles into segmentation masks
Tiles large satellite images into patches
✅ Training
Mixed precision training (AMP)
Custom loss:
Cross Entropy (Label Smoothing)
Dice Loss
Focal Loss
Boundary Loss
EMA (Exponential Moving Average)
OneCycle LR Scheduler
✅ Evaluation
Pixel Accuracy
Mean IoU (primary metric)
F1 Score, Precision, Recall
Confusion Matrix
Per-class metrics visualization
✅ Prediction
Sliding window inference
Test Time Augmentation (TTA)
Gaussian blending
Morphological post-processing
Outputs:
Segmentation maps
GeoPackage (.gpkg)
✅ Ensemble Tuning
Grid search for optimal model weights

📊 Classes
ID	Class Name
0	Background
1	RCC Roof
2	Tiled Roof
3	Tin Roof
4	Thatched Roof
5	Road
6	Waterbody
7	Transformer
8	Tank
9	Well


main(PHASE)
📈 Outputs

📊 Training curves
📉 Confusion matrix
📌 Per-class metrics
🗺️ GeoPackage files (GIS-ready)
🖼️ Visualization images

💡 Key Highlights
Handles large-scale satellite imagery
Supports multi-source datasets (CG + PB villages)
Advanced attention-based CNN + Transformer fusion
Optimized for GPU training
Produces GIS-compatible outputs

⚠️ Notes
Requires GPU for training (recommended)
Ensure correct dataset paths in CFG
Large images are automatically tiled
