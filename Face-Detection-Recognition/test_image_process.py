from app import FaceDetectionRecognitionApp

def test_image_process(image_path, method='haar'):
    app = FaceDetectionRecognitionApp(detector_method=method)
    app.process_image(image_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test Image Detection and Recognition')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--method', type=str, default='haar', choices=['haar', 'ssd'], help='Detection method')
    args = parser.parse_args()
    test_image_process(args.image, args.method)
