import cv2
import numpy as np
import sqlite3

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

def insertorupdate(Id, Name, age):
    conn = sqlite3.connect("sqlite.db")
    cmd = "SELECT * FROM USER WHERE ID="+ str(Id)
    cursor = conn.execute(cmd)
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
    if(isRecordExist == 1):
        conn.execute( "UPDATE USER SET Name=? WHERE Id=?", (Name, Id))
        conn.execute( "UPDATE USER SET age=? WHERE Id=?", (age, Id))
    else:
        conn.execute("INSERT INTO USER (Id,Name,age)values(?,?,?)", (Id, Name, age))
    conn.commit()
    conn.close()

# Insert user-defined values into table
Id = input('Enter User Id: ')
Name = input('Enter User Name: ')
age = input('Enter User Age: ')

insertorupdate(Id, Name, age)

# Detect face in web camera coding
sampleNUM = 0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        sampleNUM = sampleNUM + 1
        cv2.imwrite("dataset/user." + str(Id) + "." + str(sampleNUM) + ".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.waitKey(100)
    cv2.imshow("Face", img)
    cv2.waitKey(1)
    if(sampleNUM > 20):
        break

cam.release() 
cv2.destroyAllWindows()

