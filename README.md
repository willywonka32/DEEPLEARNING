This Python code showcases human activity recognition by leveraging a pre-trained model and MediaPipe for pose estimation. It operates by capturing frames from either a video input or a live webcam feed, and subsequently predicts activities within the frames through a deep neural network.

**Requirements:**

To execute this code, ensure you have these packages installed:

- `collections`
- `numpy`
- `opencv-python`
- `mediapipe`
- `onnx`

**Running the Code:**

To initiate the code, run the subsequent command in your terminal:

For webcam feed:
```bash
python live.py
```

For video file input:
```bash
python video.py
```

Press 'q' to exit the application.

**Code Overview:**

The `Parameters` class sets up crucial paths and constants within the code.

1. A deque named `captures` is employed to store the acquired frames.
2. The pre-trained human activity recognition model is loaded using the `cv2.dnn.readNet` function from OpenCV.
3. The MediaPipe pose estimation component is configured.
4. Frames are processed, resized, and added to the deque.
5. The code predicts activities within the captured frames using the pre-trained model.
6. Pose estimation is performed on the frames using MediaPipe.
7. Predicted activities and estimated poses are visualized on the frames.
8. Processed frames are displayed on the screen.

Feel free to reach out if you need further assistance or if you would like to improve any specific aspect of the code explanation.
