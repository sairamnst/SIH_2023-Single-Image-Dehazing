The provided file contains the following files:
     1.best.pt which is the weights for the people detection model we trained using YOLO v8 people detection algorithm and a custom hand-made dataset of more than 500 images of people in different postures and different weather conditions.
     2.camera_360.py is our combined codebase of the implementation of our idea using a 360 degree camera, which detects people beyond our field of vision and gives sound alerts if they are in the left or right of our field of vision.
     3.left.mp3, right.mp3 audio files which is used in the 360 degree camera to intimate the position of people beyond our field of vision.
     4.normal_camera.py which is the implementation of our code without a 360 degree camera but rather a normal camera which has the same functionality as camera_360.py but can't detect people beyond our field of vision.

camera_360.py is the main codebase of our idea implementation and not normal_camera.py.
normal_camera.py is just given here to emphasise on the impact that a 360 degree camera could have compared to a normal camera during fire rescue operations and extraction missions.

We have implemented the dark channel prior algorithm to dehaze the video and have rescaled every frame to increase performance of our code and also we have used numpy operations rather than general for loops to increase the performance since numpy uses c++ in the backend even if its used in python. Thus there is no interpreter operations and only direct complilation operations.

Dependencies required:
1.opencv2 (contrib files)
2.keyboard
3.playsound
4.numpy
5.cv2.ximgproc
6.ultralytics

If you need to test prerecorded videos, provide the path inside cv2.VideoCapture() function. If you want to test it on the camera, give 0 or 1 inside the function.

The code could be tested on a raspberry pi device that has linux as the operating system.
Recommended Specifications for the rasberry pi device is a minimum of 4 cores, with 1 thread each and a 8 gb RAM.
