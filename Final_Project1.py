import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import tkinter as tk
from PIL import ImageTk, Image
from threading import Thread
import pyttsx3

# Function to start/stop the hand tracking and classification process
def toggle_process():
    global running
    running = not running
    if running:
        start_button.config(text="Stop Process")
        process_thread = Thread(target=process)
        process_thread.start()
    else:
        start_button.config(text="Start Process")

# Function to handle hand tracking and classification process
def process():
    global running, word
    while running:
        success, img = cap.read()
        if not success:
            continue

        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)

            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                alphabet = labels[index]
                word += alphabet
                speak_alphabet(alphabet)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                alphabet = labels[index]
                word += alphabet
                speak_alphabet(alphabet)

            cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90,
                                                                     y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 28), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 0, 0), 2)  # Changed the color to blue
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        cv2.imshow("Image", imgOutput)
        if cv2.waitKey(1) == ord('q'):
            break

        word_label.config(text="Recognized Word: " + " ".join(word), fg="blue")  # Changed the text color to blue

    cv2.destroyAllWindows()

# Function to convert alphabet to speech
def speak_alphabet(alphabet):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.say(alphabet)
    engine.runAndWait()

# Function to clear the recognized word
def clear_word():
    global word
    word = []
    word_label.config(text="Recognized Word: ", fg="black")  # Reset the text color to black

# Create the main window
window = tk.Tk()
window.title("Sign Language and Hand Gestures Using CNN")

# Create a canvas to display the video stream
canvas = tk.Canvas(window, width=800, height=600)
canvas.pack()

# Create a start/stop button
running = False
start_button = tk.Button(window, text="Start Process", command=toggle_process)
start_button.pack()

# Create a label to display the recognized alphabet
word_label = tk.Label(window, text="Recognized Word: ", font=("Arial", 16), fg="blue")  #Set the initial text color to blue
word_label.pack()

# Create a button to clear the recognized word
clear_button = tk.Button(window, text="Clear Word", command=clear_word)
clear_button.pack()

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

# Initialize the hand detector and classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

labels = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E",
    "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y", "Z"
]

word = []

def update_frame():
    _, frame = cap.read()
    if frame is None:
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img.thumbnail((800, 600))
    img = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor=tk.NW, image=img)
    canvas.img = img
    if running:
        window.after(15, update_frame)

update_frame()
window.mainloop()
