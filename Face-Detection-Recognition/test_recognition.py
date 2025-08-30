import os
import cv2

try:
    import face_recognition
    from recognize_faces import FaceRecognizer
except ImportError:
    face_recognition = None
    FaceRecognizer = None

def test_face_recognition(image_path):
    if face_recognition is None or FaceRecognizer is None:
        print("face_recognition or dlib not installed. Skipping recognition test.")
        return

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    recognizer = FaceRecognizer()
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
        
    detector = None
    try:
        from detect_faces import FaceDetector
        detector = FaceDetector()
    except ImportError:
        print("detect_faces module not found. Skipping detection.")
        return

    if detector is None:
        print("Detector not available. Skipping test.")
        return

    face_locations = detector.detect_faces(image)
    names = recognizer.recognize_faces(image, face_locations)

    for ((x, y, w, h), name) in zip(face_locations, names):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    cv2.imshow('Face Recognition Test', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Recognized {len(names)} faces")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test Face Recognition')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    args = parser.parse_args()
    test_face_recognition(args.image)
