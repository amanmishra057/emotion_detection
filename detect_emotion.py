import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model and the Haar Cascade classifier
model = load_model('emotion_model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Dictionary to map class indices to emotion labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Start video capture from the webcam [cite: 36]
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame 
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the region of interest (ROI) - the face [cite: 38]
        roi_gray = gray_frame[y:y + h, x:x + w]
        # Resize ROI to 48x48 pixels, as required by the model [cite: 26]
        cropped_img = cv2.resize(roi_gray, (48, 48))
        
        # Prepare the image for the model
        roi_normalized = cropped_img.astype('float') / 255.0
        img_pixels = np.expand_dims(roi_normalized, axis=0)
        img_pixels = np.expand_dims(img_pixels, axis=-1)

        # Predict the emotion
        predictions = model.predict(img_pixels)
        max_index = int(np.argmax(predictions))
        predicted_emotion = emotion_dict[max_index]

        # Display the predicted emotion label on the frame [cite: 39]
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-Time Emotion Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()