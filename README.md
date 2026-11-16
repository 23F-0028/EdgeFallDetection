# EdgeFallDetection
### Edge AI-Based Human Activity Recognition and Fall Detection System

## Project Overview

EdgeFallDetection is an Edge AI-based human activity recognition and fall detection project developed using an ESP32-S3 microcontroller and MPU6050 motion sensor.

The system uses accelerometer and gyroscope readings to identify human activities through Machine Learning and Deep Learning models.

The main objective is to develop an intelligent, low-cost, and portable fall detection system that can recognize daily activities and detect potential falls.

## Project Objectives

- Collect real-time motion data using the MPU6050 sensor.
- Develop a human activity recognition dataset.
- Preprocess sensor readings using sliding windows.
- Train and evaluate Machine Learning and Deep Learning models.
- Convert trained models into TensorFlow Lite format.
- Explore deployment of optimized models on ESP32-S3.
- Support real-time fall detection and activity monitoring.

## Hardware Requirements

| Component | Purpose |
|---|---|
| ESP32-S3 | Main microcontroller |
| MPU6050 | Accelerometer and gyroscope |
| OLED Display | Display activity and system status |
| Green LED | Normal activity indication |
| Red LED | Fall alert indication |
| Buzzer | Audio alert |
| Breadboard and Jumper Wires | Circuit connections |

## Software and Technologies

- Python
- TensorFlow / Keras
- TensorFlow Lite
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Joblib
- Arduino IDE
- ESP32-S3
- Git and GitHub

## System Architecture

The project consists of the following stages:

1. Sensor Data Collection
2. Dataset Preparation
3. Data Preprocessing
4. Feature Normalization
5. Model Training
6. Model Evaluation
7. TensorFlow Lite Conversion
8. Edge Deployment

**Hardware workflow:**

MPU6050 → ESP32-S3 → Sensor Data → Activity Recognition → OLED / LED / Buzzer Alerts

**Machine Learning workflow:**

Dataset → Preprocessing → Training → Evaluation → TFLite Conversion

## Dataset Description

The project contains two separate dataset workflows.

### 1. Personal MPU6050 Dataset

Sensor readings are collected using the MPU6050 accelerometer and gyroscope.

Four activity classes are used:

- Sitting
- Standing
- Walking
- Fall

Six sensor features are extracted:

- accel_x
- accel_y
- accel_z
- gyro_x
- gyro_y
- gyro_z

### 2. External Activity Dataset

An additional external dataset is used to train and evaluate a six-class activity recognition model.

Classes:

- Falling
- Lying
- Running
- Sitting
- Standing
- Walking

The external dataset is processed separately from the personal MPU6050 dataset.

## Data Preprocessing

The personal dataset preprocessing pipeline includes:

- Missing value handling
- Timestamp validation
- Activity label encoding
- Recording session identification
- Sliding window segmentation
- Feature normalization using StandardScaler
- Session-based training and testing split
- Data leakage verification

### Preprocessing Configuration

| Parameter | Value |
|---|---|
| Window Size | 50 samples |
| Stride | 25 samples |
| Features | 6 |
| Random Seed | 42 |
| Gap Threshold | 1 second |
| Normalization | StandardScaler |

The input shape for the 1D-CNN model is:

`(50, 6)`

## Machine Learning Models

The repository contains multiple model training workflows:

- 1D Convolutional Neural Network (1D-CNN)
- Random Forest
- Support Vector Machine (SVM)

The external dataset uses a 1D-CNN architecture.

### 1D-CNN Architecture

- Input Layer: 50 × 6
- Conv1D: 32 filters, kernel size 5
- Batch Normalization
- Max Pooling
- Conv1D: 64 filters, kernel size 3
- Batch Normalization
- Global Average Pooling
- Dense Layer: 32 neurons
- Dropout: 0.3
- Output Layer: 6 classes with Softmax

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Maximum Epochs | 40 |
| Batch Size | 16 |
| Random Seed | 42 |
| Early Stopping | Patience 7 |

## Model Evaluation

The external 1D-CNN model achieved the following results on the test dataset:

### Test Performance

| Metric | Result |
|---|---:|
| Test Accuracy | 93.76% |
| Test Loss | 0.1324 |
| Training Samples | 34,018 |
| Validation Samples | 7,832 |
| Testing Samples | 8,328 |
| Epochs Completed | 8 |

### Class-wise F1 Scores

| Activity | F1 Score |
|---|---:|
| Falling | 0.94 |
| Lying | 0.96 |
| Running | 1.00 |
| Sitting | 0.75 |
| Standing | 0.81 |
| Walking | 0.99 |

Evaluation outputs include:

- Classification Report
- Confusion Matrix
- Training Accuracy and Loss Graphs
- Evaluation Metrics in JSON format

## TensorFlow Lite Optimization

The trained external CNN model is converted into a fully integer-quantized INT8 TensorFlow Lite model.

Optimization techniques include:

- TensorFlow Lite Converter
- INT8 Quantization
- Representative Dataset
- INT8 Input and Output
- Normalization Parameter Export

Generated model:

`external_activity_model_int8.tflite`

The normalization parameters are saved in:

`normalization.json`

These files support preparation for resource-efficient edge inference.

## Repository Structure

```text
EdgeFallDetection/
│
├── TFT_Test/
│
├── model_output/
├── model_output_backup/
├── processed_data/
│
├── external_model_output/
├── external_processed_data/
│
├── rf_output/
├── svm_output/
│
├── activity_dataset.csv
├── activity_dataset_consistent.csv
│
├── collect_data.py
├── collect_fall.py
├── check_dataset.py
│
├── preprocess.py
├── prepare_external_data.py
│
├── train_model.py
├── train_external_model.py
├── train_rf.py
├── train_svm.py
│
├── convert_tflite.py
├── convert_external_tflite.py
├── model_to_header.py
│
├── verify_external_tflite.py
├── README.md
│
└── confusion_matrix.png
```

## Installation and Setup

### 1. Clone Repository

```bash
git clone https://github.com/23F-0028/EdgeFallDetection.git
```

### 2. Navigate to Project

```bash
cd EdgeFallDetection
```

### 3. Create Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate Environment (Windows)

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Required Libraries

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow joblib pyserial
```

## Running the Project

### Personal Dataset Workflow

Collect sensor data:

```bash
python collect_data.py
```

Preprocess dataset:

```bash
python preprocess.py
```

Train activity recognition model:

```bash
python train_model.py
```

Convert model to TensorFlow Lite:

```bash
python convert_tflite.py
```

Generate model header:

```bash
python model_to_header.py
```

### External Dataset Workflow

Prepare external dataset:

```bash
python prepare_external_data.py
```

Train external 1D-CNN:

```bash
python train_external_model.py
```

Convert external model to INT8 TensorFlow Lite:

```bash
python convert_external_tflite.py
```

Verify converted model:

```bash
python verify_external_tflite.py
```

**Note:** External and personal models use different class mappings and must be integrated with their corresponding preprocessing and inference configurations.

## Future Improvements

- Improve sitting and standing classification.
- Increase fall detection reliability.
- Optimize model size and inference time.
- Complete ESP32-S3 edge inference integration.
- Improve real-time OLED monitoring.
- Implement reliable LED and buzzer alerts.
- Explore cloud-based activity monitoring.

## Applications

- Elderly care monitoring
- Personal safety systems
- Fall detection applications
- Wearable health monitoring
- Human activity recognition

## Developer

**Project:** EdgeFallDetection

**Domain:** Artificial Intelligence / Machine Learning / Edge AI

**Hardware:** ESP32-S3 and MPU6050

**Deep Learning:** 1D-CNN

**Framework:** TensorFlow / TensorFlow Lite

**Repository:** https://github.com/23F-0028/EdgeFallDetection

---

*This project is developed for academic and research purposes.*