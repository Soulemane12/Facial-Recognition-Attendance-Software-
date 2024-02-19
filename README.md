# Facial Recognition Attendance System

This Python script implements a facial recognition attendance system using OpenCV, NumPy, and the face_recognition library. The system captures video frames, detects faces, compares them with known faces, and marks attendance based on recognized individuals.

## Features

- **Real-time Attendance**: Automatically marks attendance by recognizing faces in real-time video streams.
- **Dynamic Tolerance**: Adjusts face recognition thresholds based on the number of faces detected for improved accuracy.
- **Attendance Logging**: Stores attendance records in a CSV file with timestamps for easy tracking.
- **Text-to-Speech Feedback**: Provides auditory feedback with a welcome message for recognized individuals.

## Dependencies

- OpenCV
- NumPy
- face_recognition
- pyttsx3 (for text-to-speech support)

## Usage

1. Install the required dependencies using `pip install -r requirements.txt`.
2. Run the script by executing `python facial_recognition_attendance.py`.
3. Specify the directory containing student images when prompted.
4. The system will start capturing video from the webcam and mark attendance in real-time.

## Example

```python
python facial_recognition_attendance.py
