## Image Classification Using a Pretrained Model

## Project Overview

This project demonstrates image classification using a pretrained deep learning model. The project uses **MobileNetV2**, a convolutional neural network pretrained on the **ImageNet** dataset, to identify objects present in an input image.

The model takes an image as input and provides the **top 5 predicted classes along with their confidence scores**.

## Objectives

* Understand the concept of pretrained deep learning models.
* Apply transfer learning for image classification.
* Load and use the MobileNetV2 pretrained model.
* Preprocess input images before prediction.
* Generate and display top-5 classification predictions.

## Technologies Used

* Python
* TensorFlow
* Keras
* MobileNetV2
* NumPy
* Pillow
* ImageNet

## Installation

Make sure Python is installed on your system.

Install the required libraries using:

bash
pip install tensorflow pillow numpy


##  How to Run

1. Open the project folder in VS Code.
2. Open the terminal.
3. Run the following command:

bash
python image_classification.py


4. Enter the complete path of the image when prompted.

Example:
F:\3194556-puppy-1903313.jpg


5. The program will display the top 5 predictions and their confidence scores.

## Model Used

### MobileNetV2

MobileNetV2 is a lightweight convolutional neural network designed for efficient image classification and computer vision applications.

In this project, the model is loaded with pretrained **ImageNet weights**, allowing it to classify images without training a new model from scratch.

python
model = MobileNetV2(weights="imagenet")


## Output

The program produces the top 5 predictions for the input image along with their confidence percentages.

Example:
Image Classification Results
1. Labrador Retriever: XX.XX%
2. Golden Retriever: XX.XX%
3. ...

## Internship Project

This project was completed as part of my **CodeOrbit Internship** in Artificial Intelligence / Machine Learning.

## GitHub Repository

https://github.com/shreelatha064-glitch/CodeOrbit_ImageClassification.git

