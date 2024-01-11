import os
import cv2
import numpy as np
import face_recognition
from gender_detection import f_my_gender
from emotion_detection import f_emotion_detection
import tensorflow as tf
from tensorflow.keras.utils import load_img
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import seaborn as sns
import warnings
from tqdm.notebook import tqdm

# instanciar detectores
gender_detector =  f_my_gender.Gender_Model()
emotion_detector = f_emotion_detection.predict_emotions()
#----------------------------------------------



def get_face_info(im):
    # face detection
    boxes_face = face_recognition.face_locations(im)
    out = []
    if len(boxes_face)!=0:
        for box_face in boxes_face:
            # segmento rostro
            box_face_fc = box_face
            x0,y1,x1,y0 = box_face
            box_face = np.array([y0,x0,y1,x1])
            face_features = {
                "gender":[],
                "emotion":[],
                "bbx_frontal_face":box_face             
            } 

            face_image = im[x0:x1,y0:y1]


            # -------------------------------------- gender_detection ---------------------------------------
            face_features["gender"] = gender_detector.predict_gender(face_image)


            # -------------------------------------- emotion_detection ---------------------------------------
            _,emotion = emotion_detector.get_emotion(im,[box_face])
            face_features["emotion"] = emotion[0]

            # -------------------------------------- out ---------------------------------------       
            out.append(face_features)
    else:
        face_features = {
            "gender":[],
            "emotion":[],
            "bbx_frontal_face":[]             
        }
        out.append(face_features)
    
    return out



def bounding_box(out, img):
    faces_detected = False
    
    for data_face in out:
        box = data_face["bbx_frontal_face"]
        if len(box) == 0:
            continue
        else:
            x0, y0, x1, y1 = box
            faces_detected = True
            img = cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
            thickness = 2
            fontSize = 0.5
            step = 20

            gender = data_face["gender"]
            emotion = data_face["emotion"]

            if gender != "No face detected":
                try:
                    cv2.putText(img, "Gender: " + gender, (x0, y0 - step - 10 * 1), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (205, 38, 38), thickness)
                except:
                    pass

            if emotion != "No face detected":
                try:
                    cv2.putText(img, "Emotion: " + emotion, (x0, y0 - step - 17 * 3), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 225, 225), thickness)
                except:
                    pass

    if not faces_detected:
        print("No faces detected in the image.")
    else:
        print("Detected Faces Information:")
        for data_face in out:
            box = data_face["bbx_frontal_face"]
            if len(box) == 0:
                continue
            else:
                gender = data_face["gender"]
                emotion = data_face["emotion"]

                print("Face Info:")
                print(f" - Gender: {gender}")
                print(f" - Emotion: {emotion}")
                print("-------------------")

    return img