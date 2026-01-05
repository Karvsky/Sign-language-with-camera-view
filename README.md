# Sign Language Detection with CNN & MediaPipe

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Status](https://img.shields.io/badge/Status-Paused-yellow)

## 📌 Project Description

This is a real-time American Sign Language (ASL) alphabet recognition application. The system operates using a two-stage architecture:
1.  **Localization & Preprocessing:** The MediaPipe library detects the hand and crops the Region of Interest (ROI), effectively resizing the image to the format required by the model.
2.  **Classification:** The processed image crop is converted to grayscale and passed to a custom Convolutional Neural Network (CNN), which recognizes one of 29 classes (A-Z, space, del, nothing).

## 📊 Results and Limitations

During the training process, the model achieved an **accuracy of 90%** on the validation set. Despite high metrics on static data, real-time performance with a live camera is inconsistent.

**Identified Issues:**
* **Dataset vs. Reality:** The training dataset consists of images taken with a uniform background and identical lighting conditions. Consequently, the CNN struggles to generalize in real-world environments (varying backgrounds, shadows, noise).
* **Task Complexity:** Many letters in the sign alphabet are visually very similar (e.g., A, E, M, N). Without depth information or broader context, a simple 2D CNN makes classification errors.
* **Project Status:** Due to the dataset limitations and the high complexity of distinguishing similar letters without advanced preprocessing (e.g., extensive augmentation, background removal), the project has been **paused**. It serves as an example of a well-trained classifier that faces challenges in deployment due to data bias.

## 💡 Technical Insights: Why YOLO would be better

The development of this project highlighted a flaw in the chosen architecture. **Currently, MediaPipe is used solely to crop and resize the hand image**, making the classification result heavily dependent on the quality of that crop and the CNN's sensitivity to the background within the crop.

A superior approach for future iterations would be to use **YOLO (You Only Look Once)** from the start:
* YOLO handles object detection and classification in a single step, considering the broader context of the image.
* It is significantly more robust to background noise compared to a standard CNN trained on cropped hands.
* It simplifies the pipeline by removing the need for a separate "crop & resize" stage.
* It eliminates the compatibility issues often faced when trying to run MediaPipe and TensorFlow in the same environment (which often requires downgrading libraries).

## 🛠️ Environment Setup

The project requires the following libraries: `tensorflow`, `opencv-python`, `mediapipe`, `numpy`.

### Windows PowerShell Setup
To correctly run the scripts in a virtual environment (venv) on Windows, follow these steps in PowerShell:

1.  Change the execution policy:
    ```powershell
    Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
    ```
2.  Activate the virtual environment:
    ```powershell
    .\venv\Scripts\activate
    ```

## 🚀 Execution Instructions (Step-by-Step)

To run the project successfully, you must follow this specific order:

### Step 1: Train and Generate the Model
First, you must create the model weights file. Run:
```bash
python main.py
