import tensorflow as tf
import numpy as np
import cv2
import os

model_path = "model.h5"
model = tf.keras.models.load_model(model_path)
class_names = ['dwayne', 'hemsworth', 'kavin']

# Load the cascade
face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')


for img_path in os.listdir("data"):
    path = os.path.join("data", img_path)

    # Read the frame
    img = cv2.imread(path)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    predict_state = False
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cropped_face = gray[y:y+h, x:x+w]
        cropped_face = cv2.resize(cropped_face, (200, 200))
        predict_state = True

        input = np.array([cropped_face.tolist()])
        predict = model.predict(input, verbose=0)
        name = class_names[np.argmax(predict)]
        img = cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                          1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display
        cv2.imshow('img', img)
        # Stop if escape key is pressed
        k = cv2.waitKey(0) & 0xff
        if k == 27:
            break
