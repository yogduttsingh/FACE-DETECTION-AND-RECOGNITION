try:
    import face_recognition
except ImportError:
    face_recognition = None

import cv2
import argparse
import os
from detect_faces import FaceDetector

if face_recognition is not None:
    from recognize_faces import FaceRecognizer
else:
    FaceRecognizer = None

class FaceDetectionRecognitionApp:
    def __init__(self, detector_method='haar', confidence_threshold=0.5):
        """
        Initialize the face detection and recognition application
        Args:
            detector_method: 'haar' or 'ssd'
            confidence_threshold: confidence threshold for SSD
        """
        self.detector = FaceDetector(detector_method)
        if FaceRecognizer is not None:
            self.recognizer = FaceRecognizer()
        else:
            self.recognizer = None
            print("Warning: face_recognition library not found. Recognition disabled.")
        self.confidence_threshold = confidence_threshold
        self.running = False

    def process_image(self, image_path):
        """
        Process a single image for face detection and recognition
        Args:
            image_path: path to the image file
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return

        # Detect faces
        faces = self.detector.detect_faces(image, self.confidence_threshold)
        
        # Recognize faces if recognizer available
        if self.recognizer is not None:
            names = self.recognizer.recognize_faces(image, faces)
        else:
            names = ["Recognition Disabled"] * len(faces)
        
        # Draw results
        for ((x, y, w, h), name) in zip(faces, names):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (0, 255, 0), 2)
        
        # Display results
        cv2.imshow('Face Detection & Recognition', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print(f"Detected {len(faces)} faces:")
        for i, name in enumerate(names):
            print(f"Face {i+1}: {name}")

    def process_video(self):
        """
        Process real-time video from webcam
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        self.running = True
        print("Press 'q' to quit, 's' to save current frame")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces
            faces = self.detector.detect_faces(frame, self.confidence_threshold)
            
            # Recognize faces if recognizer available
            if self.recognizer is not None:
                names = self.recognizer.recognize_faces(frame, faces)
            else:
                names = ["Recognition Disabled"] * len(faces)
            
            # Draw results
            for ((x, y, w, h), name) in zip(faces, names):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, (0, 255, 0), 2)

            # Display frame count and FPS
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Face Detection & Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                cv2.imwrite('captured_frame.jpg', frame)
                print("Frame saved as 'captured_frame.jpg'")

        cap.release()
        cv2.destroyAllWindows()

    def add_new_person(self, person_name, num_images=5):
        """
        Add a new person to the dataset by capturing images
        Args:
            person_name: name of the person to add
            num_images: number of images to capture
        """
        person_folder = os.path.join(os.path.dirname(__file__), 'dataset', person_name)
        os.makedirs(person_folder, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        print(f"Capturing {num_images} images for {person_name}")
        print("Press 'c' to capture, 'q' to quit")
        
        count = 0
        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces
            faces = self.detector.detect_faces(frame, self.confidence_threshold)
            
            if len(faces) == 1:
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Capture {count+1}/{num_images}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Please position one face in frame", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Add New Person', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and len(faces) == 1:
                # Save the image
                image_path = os.path.join(person_folder, f"{person_name}_{count+1}.jpg")
                cv2.imwrite(image_path, frame)
                print(f"Saved image {count+1}")
                count += 1

        cap.release()
        cv2.destroyAllWindows()
        
        if count > 0:
            print(f"Successfully added {count} images for {person_name}")
            # Reload known faces
            if self.recognizer is not None:
                self.recognizer.load_known_faces()
        else:
            print("No images were captured")

def main():
    parser = argparse.ArgumentParser(description='Face Detection & Recognition Application')
    parser.add_argument('--method', type=str, default='haar', 
                        choices=['haar', 'ssd'], help='Detection method')
    parser.add_argument('--confidence', type=float, default=0.5, 
                        help='Confidence threshold for SSD')
    parser.add_argument('--image', type=str, help='Path to image file for processing')
    parser.add_argument('--add-person', type=str, help='Add a new person to dataset')
    parser.add_argument('--num-images', type=int, default=5, 
                        help='Number of images to capture when adding a person')
    
    args = parser.parse_args()
    
    app = FaceDetectionRecognitionApp(args.method, args.confidence)
    
    if args.add_person:
        app.add_new_person(args.add_person, args.num_images)
    elif args.image:
        app.process_image(args.image)
    else:
        app.process_video()

if __name__ == "__main__":
    main()
