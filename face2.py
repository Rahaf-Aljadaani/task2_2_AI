
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Read the image
img = cv2.imread('example.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# for faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eye = eye_cascade.detectMultiScale(roi_gray)

# for eyes
for (xe, ye, we, he) in eye:
    cv2.rectangle(roi_color, (xe, ye), (xe+we, ye+he), (1, 0, 0), 2)


cv2.imshow('img', img)
cv2.waitKey()

#thank you https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
#thank you https://stackoverflow.com/questions/30508922/error-215-empty-in-function-detectmultiscale
#and thanks to every videos in youtube i watched =)
