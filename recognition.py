import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('ASL2.h5')  # Adjust path if needed
img_width, img_height =32,32  # Same as the input size of your model

# Define the labels for the 29 classes (A-Z, space, delete, nothing)
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Space', 'Delete', 'Nothing']

# Open webcam
cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define the region of interest (ROI) with a slightly larger size
    # Adjusted to be a bit bigger than the previous ROI
    x1, y1, x2, y2 = 80, 80, 350, 350  # New larger coordinates of the ROI
    roi = frame[y1:y2, x1:x2]

    # Preprocess the ROI: resize to match the model input and normalize
    roi_resized = cv2.resize(roi, (img_width, img_height))
    roi_normalized = roi_resized / 255.0
    roi_expanded = np.expand_dims(roi_normalized, axis=0)

    # Predict the hand gesture using the model
    predictions = model.predict(roi_expanded)
    predicted_class = np.argmax(predictions)
    predicted_label = labels[predicted_class]

    # Display the predicted character on the frame
    cv2.putText(frame, f'Prediction: {predicted_label}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw the larger ROI on the frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('ASL Recognition', frame)
    '''filename = f'frame_{frame_count}.png'
    cv2.imwrite(filename, roi_resized)
    frame_count += 1'''

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
