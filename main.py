# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 21:29:35 2020

@author: reeju
"""
import cv2, sys

#Load som pre-train data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
#img = cv2.imread('RDJ.png')
#imp = cv2.imread('JT.jpg')

# To capture video from webcam
webcam = cv2.VideoCapture(0)
cv2.waitKey(1)

# Iterate over Frames
while True:
    
    # Read the current Frame
    successful_frame_read, frame = webcam.read()

    # Checking the image
    #cv2.imshow("Reeju's Face Detector", frame)
    #cv2.waitKey()
    
    # Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Checking the image
    #cv2.imshow("Reeju's Face Detector", grayscaled_img)
    #cv2.waitKey()
    
    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    
    # Draw Rectangles around the face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the image with the faces spotted
    cv2.imshow("Reeju's Face Detector", frame)
    key = cv2.waitKey(1)
    
    # Stop if Q is press
    if key == 81 or key == 113:
        break

# Release the webcam object
webcam.release()

"""
# Choose an image to detect faces in
img = cv2.imread('RDJ.png')
#imp = cv2.imread('JT.jpg')

# Must convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw Rectangles around the face
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the image with the faces spotted
cv2.imshow("Reeju's Face Detector", img)
cv2.waitKey()

"""
print("Code Complete")