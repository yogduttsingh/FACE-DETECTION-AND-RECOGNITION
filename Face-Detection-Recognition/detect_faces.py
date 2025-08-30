import cv2
import numpy as np
import os

class FaceDetector:
    def __init__(self, method='haar'):
        """
        Initialize face detector with specified method
        Args:
            method: 'haar' for Haar Cascade or 'ssd' for Deep Learning SSD
        """
        self.method = method.lower()
        
        if self.method == 'haar':
            # Load Haar Cascade classifier
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
        elif self.method == 'ssd':
            # Load SSD model
            model_path = os.path.join('models', 'res10_300x300_ssd_iter_140000.caffemodel')
            config_path = os.path.join('models', 'deploy.prototxt')
            
            # Check if model files exist, if not, download them
            if not os.path.exists(model_path) or not os.path.exists(config_path):
                print("SSD model files not found. Please download them and place in models/ folder.")
                print("Download from: https://github.com/opencv/opencv_3rdparty/tree/dnn_samples_face_detector_20170830")
                self.net = None
            else:
                self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        else:
            raise ValueError("Method must be 'haar' or 'ssd'")

    def detect_faces(self, image, confidence_threshold=0.5):
        """
        Detect faces in an image
        Args:
            image: input image (numpy array)
            confidence_threshold: confidence threshold for SSD method
        Returns:
            List of bounding boxes in format (x, y, w, h)
        """
        if self.method == 'haar':
            return self._detect_haar(image)
        elif self.method == 'ssd':
            return self._detect_ssd(image, confidence_threshold)
    
    def _detect_haar(self, image):
        """Detect faces using Haar Cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def _detect_ssd(self, image, confidence_threshold):
        """Detect faces using SSD model"""
        if self.net is None:
            return []
            
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure bounding boxes stay within image dimensions
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                width = endX - startX
                height = endY - startY
                
                faces.append((startX, startY, width, height))
        
        return faces

def detect_faces_image(image_path, method='haar', confidence_threshold=0.5):
    """
    Detect faces in an image file
    Args:
        image_path: path to image file
        method: detection method ('haar' or 'ssd')
        confidence_threshold: confidence threshold for SSD
    Returns:
        image with bounding boxes and list of face coordinates
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None, []
    
    detector = FaceDetector(method)
    faces = detector.detect_faces(image, confidence_threshold)
    
    # Draw bounding boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return image, faces

def detect_faces_video(method='haar', confidence_threshold=0.5):
    """
    Real-time face detection from webcam
    Args:
        method: detection method ('haar' or 'ssd')
        confidence_threshold: confidence threshold for SSD
    """
    detector = FaceDetector(method)
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
        
        # Draw bounding boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.9, (0, 255, 0), 2)
        
        cv2.imshow('Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Detection')
    parser.add_argument('--method', type=str, default='haar', 
                       choices=['haar', 'ssd'], help='Detection method')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--confidence', type=float, default=0.5, 
                       help='Confidence threshold for SSD')
    
    args = parser.parse_args()
    
    if args.image:
        result_image, faces = detect_faces_image(args.image, args.method, args.confidence)
        if result_image is not None:
            cv2.imshow('Detected Faces', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print(f"Detected {len(faces)} faces")
    else:
        detect_faces_video(args.method, args.confidence)
