import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import numpy as np
from tkinter import Tk, filedialog, messagebox, Label, Button, Toplevel
import tkinter as tk
from PIL import Image, ImageTk
import os

# Correct model path
model_path = r"D:\Animal_Detection_Task2\animal_detection_model.keras"

# Class labels
CLASS_LABELS = [
    'Bear', 'Brown bear', 'Bull', 'Butterfly', 'Camel', 'Canary', 'Caterpillar', 'Cattle',
    'Centipede', 'Cheetah', 'Chicken', 'Crab', 'Crocodile', 'Deer', 'Duck', 'Eagle',
    'Elephant', 'Fish', 'Fox', 'Frog', 'Giraffe', 'Goat', 'Goldfish', 'Goose', 'Hamster',
    'Harbor seal', 'Hedgehog', 'Hippopotamus', 'Horse', 'Jaguar', 'Jellyfish', 'Kangaroo',
    'Koala', 'Ladybug', 'Leopard', 'Lion', 'Lizard', 'Lynx', 'Magpie', 'Monkey',
    'Moths and butterflies', 'Mouse', 'Mule', 'Ostrich', 'Otter', 'Owl', 'Panda',
    'Parrot', 'Penguin', 'Pig', 'Polar bear', 'Rabbit', 'Raccoon', 'Raven', 'Red panda',
    'Rhinoceros', 'Scorpion', 'Sea lion', 'Sea turtle', 'Seahorse', 'Shark', 'Sheep',
    'Shrimp', 'Snail', 'Snake', 'Sparrow', 'Spider', 'Squid', 'Squirrel', 'Starfish',
    'Swan', 'Tick', 'Tiger', 'Tortoise', 'Turkey', 'Turtle', 'Whale', 'Woodpecker', 'Worm', 'Zebra'
]

# Carnivorous animals for red warning
CARNIVOROUS_ANIMALS = {'Bear', 'Brown bear', 'Cheetah', 'Crocodile', 'Eagle', 'Fox', 'Jaguar',
                       'Leopard', 'Lion', 'Shark', 'Snake', 'Spider', 'Tiger', 'Wolf'}

# Load model with custom scope
try:
    with tf.keras.utils.custom_object_scope({'BatchNormalization': tf.keras.layers.BatchNormalization}):
        model = keras.models.load_model(model_path, compile=False, safe_mode=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

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

# GUI Functions
def select_image():
    img_path = filedialog.askopenfilename()
    if not img_path:
        messagebox.showinfo("Warning", "No image selected. Please choose an image.")
        return

    predicted_label, confidence_score = predict_image(img_path)

    if predicted_label == "Unknown":
        return

    # Highlight carnivorous animals in red
    if predicted_label in CARNIVOROUS_ANIMALS:
        msg = f"⚠️ Warning! Carnivorous Animal Detected ⚠️\n\nDetected Animal: {predicted_label}\nConfidence: {confidence_score}%"
        messagebox.showwarning("Prediction", msg)
    else:
        messagebox.showinfo("Prediction", f"Detected Animal: {predicted_label}\nConfidence: {confidence_score}%")

    # Display Image
    img = Image.open(img_path)
    img.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(img)

    image_label.config(image=img_tk)
    image_label.image = img_tk

def select_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if not video_path:
        messagebox.showinfo("Warning", "No video selected. Please choose a video.")
        return

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to read video file. Try a different file.")
        return

    carnivore_count = 0  # Define inside the select_video() function

    video_window = Toplevel(root)
    video_window.title("Video Detection")
    video_label = Label(video_window)
    video_label.pack()

    def update_frame():
        nonlocal carnivore_count  # Correct usage of nonlocal now
        ret, frame = cap.read()
        if not ret:
            cap.release()
            video_window.destroy()
            if carnivore_count > 0:
                messagebox.showwarning("Carnivorous Alert", f"⚠️ {carnivore_count} Carnivorous Animals Detected ⚠️")
            return

        # Preprocess frame
        img_resized = cv2.resize(frame, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0)
        img_array = preprocess_input(img_array)

        # Prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_label = CLASS_LABELS[predicted_class_index]

        # Highlight carnivorous animals
        color = (0, 0, 255) if predicted_label in CARNIVOROUS_ANIMALS else (0, 255, 0)
        if predicted_label in CARNIVOROUS_ANIMALS:
            carnivore_count += 1

        # Display prediction on video frame
        cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Convert frame for Tkinter display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        # Update Tkinter label
        video_label.imgtk = img_tk
        video_label.configure(image=img_tk)

        # Repeat for next frame
        video_label.after(10, update_frame)

    update_frame()

# Tkinter GUI
root = Tk()
root.title("Animal Detection System")

btn_select_image = Button(root, text="Select Image", command=select_image)
btn_select_image.pack(pady=10)

btn_select_video = Button(root, text="Select Video", command=select_video)
btn_select_video.pack(pady=10)

image_label = Label(root)
image_label.pack()

root.mainloop()
