# Real-Time Emotion Detection

A deep learning project that performs real-time emotion detection using a CNN model trained on facial expressions. The system can detect seven different emotions: Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised.

## Project Structure


- ├── detect_emotion.py          # Real-time emotion detection script
- ├── emotion_model.h5          # Trained CNN model
- ├── haarcascade_frontalface_default.xml    # Face detection classifier
- └── train_model.py           # Model training script


## Requirements

- Python 3.x
- OpenCV (cv2)
- TensorFlow
- NumPy

## Installation

- pip install opencv-python
- pip install tensorflow
- pip install numpy
  
### How to Get the Data
1. **Kaggle Dataset**
   - Visit [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
   - Download the dataset after creating a Kaggle account
   - Extract the downloaded file to `data/fer2013` in your project directory
   - Add the path of data(train,test) in train_model.py

## Usage

1. **Training the Model**
   To train the emotion detection model:
  
   python train_model.py
  
   This will:
   - Build a CNN architecture
   - Train the model on facial emotion dataset
   - Save the trained model as 'emotion_model.h5'

2. **Running Emotion Detection**
   To start real-time emotion detection:
   
   python detect_emotion.py
   
   - Press 'q' to quit the application

## Model Architecture

The CNN model consists of:
- Multiple Convolutional layers with ReLU activation
- MaxPooling layers for feature reduction
- Dropout layers to prevent overfitting
- Dense layers for classification
- Final Softmax layer for 7 emotion classes

## Features

- Real-time face detection using Haar Cascade Classifier
- Emotion classification using a trained CNN model
- Live webcam integration
- Visual feedback with bounding boxes and emotion labels
- Support for 7 different emotion classes

## Model Training

The model is trained using:
- Grayscale images of size 48x48 pixels
- Data augmentation for better generalization
- Adam optimizer with learning rate decay
- Categorical crossentropy loss function
- 50 epochs (configurable)
## Training Data

### Dataset
This project uses the FER2013 dataset (Facial Expression Recognition 2013) which contains:
- 35,887 grayscale images
- 48x48 pixel resolution
- 7 emotion categories




