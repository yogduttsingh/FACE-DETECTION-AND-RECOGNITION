try:
    import face_recognition
    import numpy as np
except ImportError:
    face_recognition = None
    np = None

import cv2
import os

class FaceRecognizer:
    def __init__(self, dataset_path=os.path.join(os.path.dirname(__file__), 'dataset')):
        """
        Initialize face recognizer and load known faces
        Args:
            dataset_path: path to known faces dataset folder
        """
        self.dataset_path = dataset_path
        self.known_face_encodings = []
        self.known_face_names = []
        if face_recognition is not None:
            self.load_known_faces()
        else:
            print("Warning: face_recognition library not found. Face recognition disabled.")

    def load_known_faces(self):
        """
        Load known faces from dataset folder
        """
        for person_name in os.listdir(self.dataset_path):
            person_folder = os.path.join(self.dataset_path, person_name)
            if not os.path.isdir(person_folder):
                continue
            
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    self.known_face_names.append(person_name)

    def recognize_faces(self, frame, face_locations):
        """
        Recognize faces in a frame given face locations
        Args:
            frame: image frame (numpy array)
            face_locations: list of face bounding boxes (x, y, w, h)
        Returns:
            List of names corresponding to each face location
        """
        if face_recognition is None or np is None:
            return ["Recognition Disabled"] * len(face_locations)

        rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
        names = []
        
        # Use face_recognition's own face detection instead of passing pre-detected faces
        try:
            # Let face_recognition detect faces itself
            face_locations_recognition = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations_recognition)
            
            for i, face_encoding in enumerate(face_encodings):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                    else:
                        name = "Unknown"
                else:
                    name = "Unknown"
                names.append(name)
            
            # If we have fewer encodings than faces (some faces couldn't be encoded)
            while len(names) < len(face_locations):
                names.append("Unknown")
                
        except Exception as e:
            print(f"Error in face recognition: {e}")
            # Fallback: return "Unknown" for all faces
            names = ["Unknown"] * len(face_locations)
            
        return names

def recognize_faces_image(image_path, detector_method='haar', confidence_threshold=0.5):
    """
    Recognize faces in an image file
    Args:
        image_path: path to image file
        detector_method: face detection method ('haar' or 'ssd')
        confidence_threshold: confidence threshold for SSD
    """
    import detect_faces
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    detector = detect_faces.FaceDetector(detector_method)
    recognizer = FaceRecognizer()
    
    faces = detector.detect_faces(image, confidence_threshold)
    names = recognizer.recognize_faces(image, faces)
    
    for ((x, y, w, h), name) in zip(faces, names):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 255, 0), 2)
    
    cv2.imshow('Face Recognition', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def recognize_faces_video(detector_method='haar', confidence_threshold=0.5):
    """
    Real-time face recognition from webcam
    Args:
        detector_method: face detection method ('haar' or 'ssd')
        confidence_threshold: confidence threshold for SSD
    """
    import detect_faces
    detector = detect_faces.FaceDetector(detector_method)
    recognizer = FaceRecognizer()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = detector.detect_faces(frame, confidence_threshold)
        names = recognizer.recognize_faces(frame, faces)
        
        for ((x, y, w, h), name) in zip(faces, names):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (0, 255, 0), 2)
        
        cv2.imshow('Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Recognition')
    parser.add_argument('--method', type=str, default='haar', 
                        choices=['haar', 'ssd'], help='Detection method')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--confidence', type=float, default=0.5, 
                        help='Confidence threshold for SSD')
    
    args = parser.parse_args()
    
    if args.image:
        recognize_faces_image(args.image, args.method, args.confidence)
    else:
        recognize_faces_video(args.method, args.confidence)
