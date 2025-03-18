import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import numpy as np
from tkinter import Tk, filedialog, messagebox
import tkinter as tk
from PIL import Image, ImageTk
import os

# Correct model path
model_path = r"D:\Animal_Detection_Task2\animal_detection_model.keras"

# Verify model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Error: Model file not found at {model_path}")

# Custom object scope to handle compatibility issues
try:
    with tf.keras.utils.custom_object_scope({'BatchNormalization': tf.keras.layers.BatchNormalization}):
        model = keras.models.load_model(model_path, compile=False, safe_mode=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Class labels
CLASS_LABELS = [
    'Bear', 'Brown bear', 'Bull', 'Butterfly', 'Camel', 'Canary', 'Caterpillar', 'Cattle',
    'Centipede', 'Cheetah', 'Chicken', 'Crab', 'Crocodile', 'Deer', 'Duck', 'Eagle', 'Elephant',
    'Fish', 'Fox', 'Frog', 'Giraffe', 'Goat', 'Goldfish', 'Goose', 'Hamster', 'Harbor seal',
    'Hedgehog', 'Hippopotamus', 'Horse', 'Jaguar', 'Jellyfish', 'Kangaroo', 'Koala', 'Ladybug',
    'Leopard', 'Lion', 'Lizard', 'Lynx', 'Magpie', 'Monkey', 'Moths and butterflies', 'Mouse',
    'Mule', 'Ostrich', 'Otter', 'Owl', 'Panda', 'Parrot', 'Penguin', 'Pig', 'Polar bear', 'Rabbit',
    'Raccoon', 'Raven', 'Red panda', 'Rhinoceros', 'Scorpion', 'Sea lion', 'Sea turtle', 'Seahorse',
    'Shark', 'Sheep', 'Shrimp', 'Snail', 'Snake', 'Sparrow', 'Spider', 'Squid', 'Squirrel',
    'Starfish', 'Swan', 'Tick', 'Tiger', 'Tortoise', 'Turkey', 'Turtle', 'Whale', 'Woodpecker',
    'Worm', 'Zebra'
]

# Carnivorous animals for alert feature
CARNIVOROUS_ANIMALS = {'Bear', 'Brown bear', 'Cheetah', 'Crocodile', 'Eagle', 'Fox', 'Jaguar', 'Leopard', 'Lion', 'Shark', 'Snake', 'Tiger', 'Wolf'}

# Image preprocessing function
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        messagebox.showerror("Error", "Invalid image file. Please select a valid image.")
        return None

    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Prediction function
def predict_image(img_path):
    img_array = preprocess_image(img_path)
    if img_array is None:
        return "Unknown", 0.0

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_label = CLASS_LABELS[predicted_class_index]
    confidence_score = round(predictions[0][predicted_class_index] * 100, 2)

    return predicted_label, confidence_score

# Image detection function
def select_image():
    img_path = filedialog.askopenfilename()
    if not img_path:
        messagebox.showinfo("Warning", "No image selected. Please choose an image.")
        return

    predicted_label, confidence_score = predict_image(img_path)
    if predicted_label == "Unknown":
        return

    if predicted_label in CARNIVOROUS_ANIMALS:
        messagebox.showwarning("Alert", f" **CARNIVOROUS ANIMAL DETECTED!**\n{predicted_label} ({confidence_score}%)")

    messagebox.showinfo("Prediction", f"Detected Animal: {predicted_label}\nConfidence: {confidence_score}%")

    img = Image.open(img_path)
    img.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(img)

    image_label.config(image=img_tk)
    image_label.image = img_tk

# Clear image and result
def clear_image():
    image_label.config(image="")
    messagebox.showinfo("Info", "Image cleared successfully!")

# Tkinter GUI
root = Tk()
root.title("Animal Detection System")

# Layout Configuration
btn_select_image = tk.Button(root, text="Select Image", command=select_image, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
btn_select_image.pack(pady=10)

btn_clear = tk.Button(root, text="Clear Image", command=clear_image, bg="#f44336", fg="white", font=("Arial", 12, "bold"))
btn_clear.pack(pady=5)

image_label = tk.Label(root)
image_label.pack(pady=10)

root.mainloop()
