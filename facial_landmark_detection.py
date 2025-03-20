import dlib
import cv2
import os
from imutils import face_utils
import scipy.io

detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

image = cv2.imread("/Users/edelta076/Desktop/Project_VID_Assistant/face_images/fimg1.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = detector(gray)
for face in faces:
    landmarks = predictor(gray, face)
    landmarks = face_utils.shape_to_np(landmarks)
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
        resized_image = cv2.resize(image, (412, 312))  
        normalize_image = resized_image / 255.0

output_folder = "preprocessed_2D"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_image_path = os.path.join(output_folder, "fimg1.png")
cv2.imwrite(output_image_path, normalize_image)
cv2.imshow("Landmarks", normalize_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#print(f"Final image saved at: {output_image_path}")


# Load the .mat file (this contains the landmarks)
mat_file = scipy.io.loadmat('/Users/edelta076/Desktop/Project_VID_Assistant/AFLW2000/image00002.mat')
print(mat_file.keys())

landmarks_2d = mat_file['pt2d']

# Extract the 3D landmarks (pt3d_68)
landmarks_3d = mat_file['pt3d_68']

# Print to inspect the shape and structure
print(landmarks_2d.shape)
print(landmarks_3d.shape)
#landmarks = mat_file['landmarks'][0]  # Extract the landmark positions

# Load the corresponding image
#image = cv2.imread('/Users/edelta076/Desktop/Project_VID_Assistant/AFLW2000/image00002.jpg')




