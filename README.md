🔷 Cube Shatter Effect

A real-time 3D cube shatter / particle explosion effect built with Python computer vision.
This project simulates a futuristic AR-style interaction where a cube visually breaks apart into dynamic particles and reacts in real time using webcam input.

🚀 Overview

This project uses Python + OpenCV + MediaPipe to track the user’s hand and trigger a visually engaging cube shatter animation effect.

The goal is to combine:

🎯 Real-time hand tracking
🧠 Computer vision processing
🎨 Generative visual effects (particle / cube explosion)
🖥️ Live webcam interaction

The result is a smooth, interactive AR-style visual system.

✨ Features
🖐️ Real-time hand detection using MediaPipe
🎥 Live webcam input processing
💥 Dynamic cube shatter / explosion effect
🧊 Particle-based visual system
⚡ Smooth frame rendering with OpenCV
🧠 Gesture-based interaction system (expandable)
🧰 Tech Stack
Python 🐍
OpenCV 👁️
MediaPipe ✋
NumPy 🔢

example:
Cube reacts to hand movement → shatters into particles → rebuilds dynamically
🛠️ How It Works
The webcam captures live video input
MediaPipe detects hand landmarks in real time
Landmark coordinates are mapped into screen space
When interaction is detected (hand movement / trigger point):
A cube object is rendered
It breaks into multiple particles
Particles spread dynamically using physics-like motion
The effect updates every frame for smooth animation
📦 Installation
git clone https://github.com/Nikushhaa/Cube-shatter-effect.git
cd Cube-shatter-effect
pip install opencv-python mediapipe numpy
python main.py
🎮 Controls
✋ Move hand → interact with cube
🫳 Open palm → trigger effect (if implemented)
👆 Finger movement → influence particle direction
🧠 Concepts Used
Computer Vision
Real-time coordinate mapping
Particle systems
Gesture recognition
Frame-by-frame rendering pipeline
📈 Future Improvements
Add 3D cube rendering (OpenGL / PyOpenGL)
Gesture-based controls (pinch, grab, rotate)
Sound effects for shatter
TouchDesigner / shader integration
Physics-based particle simulation
📁 Project Structure
Cube-shatter-effect/
│
├── main.py
├── hand_tracking.py
├── effects.py
├── utils.py
└── README.md
🔥 Why this project is cool

This project demonstrates how computer vision can be used for real-time generative visuals, turning simple webcam input into an interactive AR-style experience.

📜 License

MIT License — feel free to use and modify.

👨‍💻 Author

Made by Nikoloz Nakashidze
