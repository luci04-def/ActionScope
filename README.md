# Video Action Recognition System

## Overview
This project implements a **video action recognition system** that detects and classifies human actions from video clips.  
Instead of training a model from scratch, the system uses a **pretrained deep learning model** to recognize actions based on motion patterns across video frames.

The focus of the project is on building a **reliable video analysis and inference pipeline**, suitable for hackathons and academic demonstrations.

---

## Problem Statement
Understanding human actions from videos is a key challenge in computer vision, especially for applications such as:
- Surveillance systems  
- Sports analytics  
- Human–computer interaction  

Image-based approaches fail to capture **temporal motion information**, which is essential for accurate action recognition.  
This project aims to analyze video data and correctly classify the action being performed using a robust and scalable approach.

---

## Proposed Solution
The system uses a **pretrained 3D Convolutional Neural Network (R3D-18)** for video action recognition.

### Key points:
- The model is already trained on large benchmark datasets  
- No training or fine-tuning is performed  
- The model is used **only for inference**

### Pipeline:
1. Read video input  
2. Extract frames from the video  
3. Prepare frame sequences  
4. Run inference using the pretrained model  
5. Output the predicted action with confidence  

---

## Model Used
**R3D-18 (3D ResNet-18)**  
- Type: Pretrained video action recognition model  
- Purpose: Classify human actions based on motion across video frames  

The model outputs probability scores for predefined action classes.  
The action with the highest probability is selected as the final prediction.

---

## Datasets
- **UCF101** – Primary dataset  
- **HMDB51** – Secondary dataset for additional testing  

The datasets are used **only for evaluation and demonstration**, not for training the model.

---

## System Architecture

The system follows an inference-only video action recognition pipeline using a pretrained deep learning model.

**Pipeline Flow:**

Input Video  
↓  
Video Loading  
↓  
Frame Extraction  
↓  
Temporal Frame Sampling  
↓  
Frame Preprocessing (Resize, Normalize)  
↓  
Pretrained R3D-18 Model  
↓  
Action Probability Scores  
↓  
Final Action Prediction  

The model weights remain fixed during execution.  
No training or fine-tuning is performed.

---


## Technologies Used
- Python  
- PyTorch  
- Torchvision  
- OpenCV  
- Google Colab  

---

## Key Features
- Video-based human action recognition  
- Uses temporal information from video frames  
- No model training required  
- Lightweight and scalable inference pipeline  
- Supports benchmark datasets  

---

## How It Works
1. A video file is provided as input  
2. Frames are extracted and resized  
3. A fixed number of frames is selected to form a video clip  
4. The clip is passed to the pretrained model  
5. The model outputs action probabilities  
6. The most probable action is reported as the final result  

---

## Development Methodology
The project is developed over a **7-day period** using an incremental approach:

- **Day 1:** Dataset understanding and action selection  
- **Day 2–3:** Video loading and frame extraction  
- **Day 4:** Model inference integration  
- **Day 5–6:** Testing and refinement  
- **Day 7:** Documentation and final submission  

The emphasis is on **correctness, explainability, and reliability**.

---

## Limitations
- The system can recognize only actions supported by the pretrained model  
- Accuracy depends on video quality and camera viewpoint  
- Real-time performance is not the primary focus  

---

## Conclusion
This project demonstrates how **pretrained deep learning models** can be effectively used for video action recognition without training from scratch.  
By focusing on video preprocessing and inference pipeline design, the system provides accurate and interpretable predictions suitable for real-world applications.

---

## Future Enhancements
- Real-time action detection  
- Support for additional action classes  
- Improved visualization of predictions  
- Web-based user interface  


## Progress

- Day 1: Problem understanding and dataset selection
- Day 2: GitHub repository setup and project structure
- Day 3: Video loading, frame extraction, and temporal sampling

