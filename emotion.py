import cv2
import numpy as np
import os
import sqlite3
import tensorflow as tf

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer/trainingdata.yml")

def get_profile(id):
    conn = sqlite3.connect("sqlite.db")
    cursor = conn.execute("SELECT * FROM USER WHERE id=?", (str(id),))
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile

def detect_emotion(face_roi_color):
    # Load the emotion detection model
    emotion_model = tf.keras.models.load_model("emotion_detection_model.h5")
    
    # Preprocess input
    face_roi_gray = cv2.cvtColor(face_roi_color, cv2.COLOR_BGR2GRAY)
    face_roi_gray = cv2.resize(face_roi_gray, (48, 48))
    face_roi_gray = face_roi_gray / 255.0
    face_roi_gray = np.reshape(face_roi_gray, (1, 48, 48, 1))

    # Predict emotion
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    emotion_prediction = emotion_model.predict(face_roi_gray)
    emotion_label_arg = np.argmax(emotion_prediction)
    emotion_text = emotion_labels[emotion_label_arg]

    return emotion_text

cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        id, conf = recognizer.predict(roi_gray)
        profile = get_profile(id)
        
        if profile is not None:
            cv2.putText(img, "Name: "+str(profile[1]), (x, y+h+20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 127), 2)
            cv2.putText(img, "Age: "+str(profile[2]), (x, y+h+40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 127), 2)

        # Detect emotion
        face_roi_color = img[y:y+h, x:x+w]
        detected_emotion = detect_emotion(face_roi_color)
        cv2.putText(img, "Emotion: "+detected_emotion, (x, y+h+60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 127), 2)

    cv2.imshow("FACE", img)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
