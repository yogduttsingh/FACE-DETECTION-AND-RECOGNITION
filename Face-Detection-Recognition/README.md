# Face Detection and Recognition System

A comprehensive Python-based face detection and recognition system that supports both image processing and real-time video analysis.

## Features

- **Face Detection**: Supports both Haar Cascade and Deep Learning (SSD) methods
- **Face Recognition**: Uses face_recognition library for accurate face matching
- **Real-time Processing**: Works with webcam for live face detection and recognition
- **Dataset Management**: Easy addition of new faces to the recognition database
- **Modular Design**: Separate modules for detection, recognition, and application

## Project Structure

```
Face-Detection-Recognition/
│── dataset/              # Store known faces (person1/, person2/, etc.)
│── models/               # Pre-trained models (download required for SSD)
│── detect_faces.py       # Face detection module
│── recognize_faces.py    # Face recognition module
│── app.py               # Main application
│── requirements.txt     # Python dependencies
│── README.md           # This file
```

## Installation

1. **Clone or download this project**
   ```bash
   cd Face-Detection-Recognition
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download SSD model files (optional, for deep learning detection)**
   - Download from: https://github.com/opencv/opencv_3rdparty/tree/dnn_samples_face_detector_20170830
   - Place `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` in the `models/` folder

## Usage

### 1. Face Detection Only

**Detect faces in an image:**
```bash
python detect_faces.py --image path/to/image.jpg --method haar
python detect_faces.py --image path/to/image.jpg --method ssd --confidence 0.6
```

**Real-time face detection from webcam:**
```bash
python detect_faces.py --method haar
python detect_faces.py --method ssd --confidence 0.6
```

### 2. Face Recognition Only

**Recognize faces in an image:**
```bash
python recognize_faces.py --image path/to/image.jpg --method haar
```

**Real-time face recognition from webcam:**
```bash
python recognize_faces.py --method haar
```

### 3. Complete Application (Detection + Recognition)

**Process an image:**
```bash
python app.py --image path/to/image.jpg --method haar
```

**Real-time processing from webcam:**
```bash
python app.py --method haar
```

**Add a new person to the dataset:**
```bash
python app.py --add-person "John Doe" --num-images 5 --method haar
```

## Dataset Structure

The `dataset/` folder should contain subfolders for each person:
```
dataset/
├── person1/
│   ├── person1_1.jpg
│   ├── person1_2.jpg
│   └── person1_3.jpg
├── person2/
│   ├── person2_1.jpg
│   └── person2_2.jpg
└── ...
```

Each subfolder should contain multiple images of the same person for better recognition accuracy.

## Command Line Arguments

### Common Arguments:
- `--method`: Detection method (`haar` or `ssd`), default: `haar`
- `--confidence`: Confidence threshold for SSD method (0.0-1.0), default: 0.5
- `--image`: Path to image file for processing

### Application-specific Arguments:
- `--add-person`: Name of person to add to dataset
- `--num-images`: Number of images to capture when adding a person, default: 5

## Detection Methods

### 1. Haar Cascade
- **Pros**: Fast, lightweight, works well for frontal faces
- **Cons**: Less accurate for non-frontal faces and complex backgrounds

### 2. SSD (Deep Learning)
- **Pros**: More accurate, better at handling various angles and lighting
- **Cons**: Requires model files, slightly slower, needs more computational power

## Tips for Better Recognition

1. **Good Lighting**: Ensure faces are well-lit
2. **Frontal Faces**: Capture faces looking directly at the camera
3. **Multiple Images**: Add 3-5 images per person for better accuracy
4. **Consistent Environment**: Use similar lighting and background conditions
5. **High Resolution**: Use clear, high-quality images

## Troubleshooting

### Common Issues:

1. **"No module named 'face_recognition'**
   - Make sure dlib is properly installed
   - Try: `pip install cmake` first, then `pip install dlib face-recognition`

2. **Webcam not working**
   - Check if webcam is connected and accessible
   - Try different camera index: modify `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

3. **Poor recognition accuracy**
   - Add more training images
   - Ensure good lighting conditions
   - Use frontal face images

4. **SSD model not found**
   - Download the required model files and place in `models/` folder

## Dependencies

- Python >= 3.8
- OpenCV (opencv-python)
- NumPy
- face-recognition
- dlib
- TensorFlow (optional for advanced features)

## Performance Notes

- **Haar Cascade**: ~15-30 FPS on average hardware
- **SSD**: ~5-15 FPS on average hardware (with GPU acceleration recommended)
- Recognition accuracy improves with more training images per person

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues and enhancement requests!
