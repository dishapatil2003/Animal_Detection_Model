# Animal Detection Model
Description
The Animal Detection Model is a machine learning-based model that detects and classifies animals in images and videos. It is designed to identify 80 different species of animals, including both common and rare animals. The model is trained on a dataset of images and provides a GUI interface to display the detected animals.

Key Features:
Detects and classifies 80 different animal species.
Highlights carnivorous animals in red with a pop-up showing their count.
GUI for easy interaction with image/video preview.
Built using machine learning techniques to recognize animals in various environments.
Technologies Used:
Python - Programming language.
TensorFlow/Keras - Deep learning framework for building the model.
OpenCV - For image and video processing.
Tkinter - For creating the graphical user interface (GUI).
Git - Version control.
Dataset:
The model is trained on a dataset containing:

18,083 training images
4,483 validation images
6,505 test images
The dataset includes images of animals like Bear, Butterfly, Cheetah, Dolphin, Elephant, Zebra, etc.

How It Works:
The model is trained using the TensorFlow framework.
The model uses Convolutional Neural Networks (CNNs) to detect features from images.
It classifies animals into predefined categories and identifies if they are carnivorous (highlighting them in red).
The system then uses a GUI to allow the user to interact with the application, preview images/videos, and see detection results.
