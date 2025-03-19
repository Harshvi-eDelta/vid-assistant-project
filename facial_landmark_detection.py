import cv2
import dlib
from imutils import face_utils
import matplotlib.pyplot as plt

# Load Dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/edelta076/Desktop/Project_VID_Assistant/shape_predictor_68_face_landmarks.dat")  # Download from Dlib

# Load the image
image = cv2.imread("/Users/edelta076/Desktop/Project_VID_Assistant/face_images/fimg5.png")  
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = detector(gray)

for face in faces:
    # Get facial landmarks
    landmarks = predictor(gray, face)
    landmarks = face_utils.shape_to_np(landmarks)  # Convert to numpy array
    # Draw circles around landmarks
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

# Show the image with detected landmarks
cv2.imshow("Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
