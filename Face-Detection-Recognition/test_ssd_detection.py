import cv2
import os
from detect_faces import FaceDetector

def test_ssd_detection(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    detector = FaceDetector(method='ssd')
    image = cv2.imread(image_path)
    faces = detector.detect_faces(image, confidence_threshold=0.5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('SSD Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Detected {len(faces)} faces using SSD")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test SSD Face Detection')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    args = parser.parse_args()
    test_ssd_detection(args.image)
