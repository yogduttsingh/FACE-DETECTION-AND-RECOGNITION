import sys
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    if sys.version_info >= (3, 8):
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} - OK")
        return True
    else:
        print(f"✗ Python version {sys.version_info.major}.{sys.version_info.minor} is too old. Requires Python 3.8+")
        return False

def check_package(package_name):
    """Check if a package is installed"""
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is not None:
            print(f"✓ {package_name} - OK")
            return True
        else:
            print(f"✗ {package_name} - Not installed")
            return False
    except ImportError:
        print(f"✗ {package_name} - Not installed")
        return False

def check_opencv():
    """Check OpenCV installation and version"""
    try:
        import cv2
        version = cv2.__version__
        print(f"✓ OpenCV {version} - OK")
        return True
    except ImportError:
        print("✗ OpenCV - Not installed")
        return False

def check_face_recognition():
    """Check face_recognition installation"""
    try:
        import face_recognition
        print("✓ face_recognition - OK")
        return True
    except ImportError as e:
        print(f"✗ face_recognition - Error: {e}")
        return False

def check_dlib():
    """Check dlib installation"""
    try:
        import dlib
        print(f"✓ dlib {dlib.__version__} - OK")
        return True
    except ImportError as e:
        print(f"✗ dlib - Error: {e}")
        return False

def check_models():
    """Check if model files exist"""
    import os
    models_dir = 'models'
    required_files = [
        'deploy.prototxt',
        'res10_300x300_ssd_iter_140000.caffemodel'
    ]
    
    print("Checking model files...")
    all_exist = True
    
    for filename in required_files:
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            print(f"✓ {filename} - Found")
        else:
            print(f"✗ {filename} - Missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all installation checks"""
    print("=" * 50)
    print("Face Detection & Recognition - Installation Check")
    print("=" * 50)
    
    success = True
    
    # Check Python version
    success &= check_python_version()
    print()
    
    # Check required packages
    print("Checking required packages...")
    success &= check_opencv()
    success &= check_dlib()
    success &= check_face_recognition()
    success &= check_package('numpy')
    print()
    
    # Check optional packages
    print("Checking optional packages...")
    tensorflow_ok = check_package('tensorflow')
    print()
    
    # Check model files
    models_ok = check_models()
    print()
    
    print("=" * 50)
    if success:
        print("✓ All required components are installed correctly!")
        print("\nYou can now run:")
        print("  python detect_faces.py --method haar")
        print("  python recognize_faces.py --method haar")
        print("  python app.py --method haar")
    else:
        print("✗ Some components are missing or have issues.")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
        
        if not models_ok:
            print("\nDownload SSD model files:")
            print("  python download_models.py --download")
            print("  or download manually from:")
            print("  https://github.com/opencv/opencv_3rdparty/tree/dnn_samples_face_detector_20170830")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
