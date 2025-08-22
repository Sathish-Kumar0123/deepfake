# Deepfake Detection using CNN
## Project Overview
This project focuses on detecting **Deepfake images** using a **Convolutional Neural Network (CNN)**.
The model is trained on a dataset of real and fake faces and achieves an **accuracy of 70%**.

## Features

* Preprocessing of image dataset (real vs fake).
* Training CNN model with TensorFlow/Keras.
* Evaluation of model with accuracy and loss metrics.
* Predicting whether a new image is **real** or **fake**.

## Installation & Requirements

1. Clone the repository
   git clone https://github.com/your-username/deepfake-detection.git
   cd deepfake-detectio
2. Install dependencies
   pip install -r requirements.txt

##  Model Training

* The CNN model was trained on a dataset of **real and deepfake faces**.
* Optimizer: `Adam`
* Loss: `binary_crossentropy`
* Epochs: 20 (can be tuned)
* Final Accuracy: **70%**
## 🖼️ How to Use

* Train the model:
  python src/train.py
* Test the model:
  python src/test.py
* Predict on a new image:
  python src/predict.py --image path/to/image.jpg
## Results

* Training Accuracy: \~70%
* Validation Accuracy: \~68%
* The model successfully detects deepfake images but can be further improved with larger datasets and advanced architectures.

## Future Improvements

* Use advanced models like **EfficientNet, XceptionNet, or ResNet50**.
* Experiment with larger datasets.
* Apply data augmentation for better generalization.
* Deploy the model as a **Web App** using Flask/Streamlit.

