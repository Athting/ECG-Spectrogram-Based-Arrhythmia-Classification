## ECG Signal-to-Image Arrhythmia Classification
[Signal-to-Image-Translation-Model-for-ECG-Based-Heartbeat-Classification.pptx](https://github.com/user-attachments/files/24339843/Signal-to-Image-Translation-Model-for-ECG-Based-Heartbeat-Classification.pptx)

This project presents a hybrid deep learning pipeline that converts 1D ECG signals into 2D spectrogram images and classifies heartbeats using a Convolutional Neural Network (CNN).

## 🔍 Problem Statement
Manual ECG interpretation is time-consuming and prone to bias. Traditional 1D signal-based models struggle with noise and interpretability. This project leverages signal-to-image translation to improve robustness and clinical insight.

## 🧠 Approach
- **Signal Processing**: Bandpass filtering (0.5–40 Hz), R-peak detection, normalization
- **Signal-to-Image Translation**: STFT-based spectrogram generation
- **Deep Learning**: CNN trained on spectrogram images
- **Explainability**: Grad-CAM visualization for model interpretability

## 🗂 Dataset
- **MIT-BIH Arrhythmia Database**
- Sampling Rate: 360 Hz
- Patient-wise split: 70% Train / 15% Validation / 15% Test

## 🏗 Model Architecture
- Input: 128×128×3 spectrogram images
- Conv2D + ReLU + BatchNorm + MaxPooling (×3)
- Dense(128) + Dropout(0.5)
- Softmax Output (3 classes)


## 📊 Results
| Metric | Value |
|------|------|
| Test Accuracy | **86.18%** |
| Ventricular Recall | **0.93** |
| Macro F1-score | **0.60** |

The model prioritizes detection of life-threatening ventricular arrhythmias while maintaining strong generalization on unseen patient data.

## 🔬 Explainability
Grad-CAM heatmaps highlight frequency–time regions critical to classification, ensuring biologically interpretable predictions suitable for clinical settings.

## 🚀 Future Work
- Advanced data balancing for minority classes
- Real-time wearable ECG integration
- Extension to EEG and EMG signal analysis

## 🛠 Tech Stack
Python, TensorFlow/Keras, NumPy, Librosa, OpenCV, Scikit-learn, Matplotlib

## 📌 Acknowledgment
Course Project — AI & ML  
IIIT Allahabad  
Guided by **Dr. Shanti Chandra**

[Uploading Signal-to-Image-Translation-Model-for-ECG-Based-Heartbeat-Classification.pptx…]()

