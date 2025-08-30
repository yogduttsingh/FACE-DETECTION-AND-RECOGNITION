import face_recognition
import cv2
import os

# Test basic face recognition functionality
def test_basic_face_recognition():
    print("Testing basic face recognition...")
    
    # Fix image path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "dataset", "Test User", "Test User_1.jpg")
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Load image with face_recognition
    image = face_recognition.load_image_file(image_path)
    print(f"Image loaded successfully: {image.shape}")
    
    # Find all face locations
    face_locations = face_recognition.face_locations(image)
    print(f"Found {len(face_locations)} face(s) in the image")
    
    # Get face encodings
    face_encodings = face_recognition.face_encodings(image, face_locations)
    print(f"Generated {len(face_encodings)} face encoding(s)")
    
    # Display the image with bounding boxes
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
    
    cv2.imshow('Face Detection Test', image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Test completed successfully!")

if __name__ == "__main__":
    test_basic_face_recognition()
