import os
import urllib.request
import argparse

def download_ssd_models():
    """
    Download SSD model files for face detection
    """
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # URLs for the model files
    model_urls = {
        'deploy.prototxt': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
        'res10_300x300_ssd_iter_140000.caffemodel': 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel'
    }
    
    print("Downloading SSD model files...")
    
    for filename, url in model_urls.items():
        filepath = os.path.join(models_dir, filename)
        
        if os.path.exists(filepath):
            print(f"{filename} already exists. Skipping download.")
            continue
            
        try:
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            print("Please download manually from:")
            print("https://github.com/opencv/opencv_3rdparty/tree/dnn_samples_face_detector_20170830")
            return False
    
    print("All model files downloaded successfully!")
    return True

def check_models():
    """
    Check if required model files exist
    """
    models_dir = 'models'
    required_files = [
        'deploy.prototxt',
        'res10_300x300_ssd_iter_140000.caffemodel'
    ]
    
    missing_files = []
    
    for filename in required_files:
        filepath = os.path.join(models_dir, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
    
    if missing_files:
        print("Missing model files:")
        for filename in missing_files:
            print(f"  - {filename}")
        return False
    else:
        print("All model files are present.")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manage SSD model files')
    parser.add_argument('--download', action='store_true', help='Download model files')
    parser.add_argument('--check', action='store_true', help='Check if model files exist')
    
    args = parser.parse_args()
    
    if args.download:
        download_ssd_models()
    elif args.check:
        check_models()
    else:
        print("Please specify an action: --download or --check")
        print("Usage:")
        print("  python download_models.py --download  # Download model files")
        print("  python download_models.py --check     # Check if model files exist")
