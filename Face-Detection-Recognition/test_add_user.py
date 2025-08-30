from app import FaceDetectionRecognitionApp

def test_add_user(person_name, num_images=3):
    app = FaceDetectionRecognitionApp()
    app.add_new_person(person_name, num_images)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test Add New User')
    parser.add_argument('--name', type=str, required=True, help='Name of the person to add')
    parser.add_argument('--num', type=int, default=3, help='Number of images to capture')
    args = parser.parse_args()
    test_add_user(args.name, args.num)
