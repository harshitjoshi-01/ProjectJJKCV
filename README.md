# Project JJKCV

A real-time **Computer Vision** project inspired by Jujutsu Kaisen, built using Python, OpenCV and MediaPipe.

The application recognizes custom hand gestures through a webcam and activates different cursed techniques by playing animated visual effects. 
The project is designed around a modular engine where every technique is implemented independently while sharing the same animation and rendering pipeline.


# Features

* Real-time hand tracking
* Gesture recognition using MediaPipe landmarks
* Modular technique system
* Smooth animation playback
* Automatic animation looping
* Fade-in transitions
* Gesture normalization for different hand sizes
* Easy to add new techniques
* Live webcam rendering


# Current Techniques

## Infinite Void

Gesture Requirements

* Index finger extended
* Middle finger extended
* Ring finger folded
* Pinky folded
* Index and middle fingers crossed

Animation Behaviour

* Plays intro animation
* Automatically loops the ending section
* Smooth fade-in
* Exits after the configured delay

---

##  Reversal Red

Gesture Requirements

* Index finger extended
* Middle finger folded
* Ring finger folded
* Pinky folded

Animation Behaviour

* Plays the Red animation
* Loops near the ending
* Automatic exit timer

---

## Lapse Blue

Gesture Requirements

* All fingers extended
* Thumb extended
* Fingers spread apart

Animation Behaviour

* Plays Blue animation
* Supports looping
* Uses spread-distance calculations for accurate recognition

---

# Project Architecture

```
ProjectJJKCV
│
├── HandTrackingModule.py
├── jujutsu_engine.py
├── assets/
│     ├── videos/
│     ├── effects/
│     └── images/
│
├── main.py
├── requirements.txt
└── README.md
```

---

# How It Works

### 1. Webcam Capture

OpenCV continuously captures frames from the webcam.

↓

### 2. Hand Detection

MediaPipe detects all 21 hand landmarks.

↓

### 3. Gesture Recognition

Each registered technique independently checks whether its gesture conditions are satisfied.

↓

### 4. Technique Activation

The engine activates the matching technique and resets any previously active animation.

↓

### 5. Animation Rendering

The selected animation is blended with the background using OpenCV.

↓

### 6. Display

The webcam feed is displayed together with the active cursed technique.

---

# Design

The engine follows an object-oriented design.

Every cursed technique inherits from a common base class.

Each technique defines:

* Its own gesture logic
* Animation file
* Loop duration
* Exit delay

Because of this design, adding a new cursed technique only requires creating another subclass instead of modifying the entire engine.

---

# Technologies Used

* Python
* OpenCV
* MediaPipe
* NumPy

---

# Installation

Clone the repository

```bash
git clone https://github.com/harshitjoshi-01/ProjectJJKCV.git
```

Go into the project

```bash
cd ProjectJJKCV
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python main.py
```

---

# Future Improvements

* Hollow Purple
* Particle effects
* Sound effect
---

# What I Learned

This project helped me gain practical experience in:

* Computer Vision
* OpenCV
* MediaPipe
* Hand Landmark Detection
* Gesture Recognition
* State Management
* Object-Oriented Programming
* Real-Time Video Processing
* Animation Blending
* Python Project Architecture

---

# Disclaimer

This is a fan-made educational project inspired by **Jujutsu Kaisen**.

All names and concepts belong to their respective copyright owners.

This project was created solely for learning computer vision and software development.

---

# Author

**Harshit Joshi**

Python • Computer Vision • AI • OpenCV

GitHub:
https://github.com/harshitjoshi-01
