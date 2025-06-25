import cv2
import os
import numpy as np
from PIL import Image
from threading import Thread
import time

def getImagesAndLabels(path):
    # path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    # empty ID list
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


def TrainImages():
    # Create required directories if they don't exist
    if not os.path.exists("TrainingImageLabel"):
        os.makedirs("TrainingImageLabel")
    if not os.path.exists("TrainingImage"):
        os.makedirs("TrainingImage")
        
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Cascade file not found at: {cascade_path}")
    
    detector = cv2.CascadeClassifier(cascade_path)
    faces, Id = getImagesAndLabels("TrainingImage")
    
    # Create threads properly
    train_thread = Thread(target=recognizer.train, args=(faces, np.array(Id)))
    count_thread = Thread(target=counter_img, args=("TrainingImage",))
    
    train_thread.start()
    count_thread.start()
    
    train_thread.join()
    count_thread.join()
    
    recognizer.save("TrainingImageLabel"+os.sep+"Trainner.yml")
    print("\nAll Images Trained Successfully")


def counter_img(path):
    imgcounter = 1
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        print(str(imgcounter) + " Images Trained", end="\r")
        time.sleep(0.008)
        imgcounter += 1