# RPPG Detection of Heart Rate through Face and FingerTip

## 1. Introduction
### Purpose
This model was developed to estimate heart rate remotely using facial video analysis through green channel intensity variations, leveraging the principles of rPPG.  

### Scope
- Limited to estimating heart rate from pre-recorded or live videos of faces in well-lit conditions.
- Does not handle multiple faces or significant occlusions.

### Audience
Developers, researchers, and healthcare professionals interested in non-contact heart rate monitoring.

---

## 2. Overview of the Model
### Description
The model processes video frames to detect a face, extract the green channel's intensity, filter the signal, and estimate heart rate using FFT.  

### Diagram
1. **Video Feed** →  
2. **Face Detection** →  
3. **Green Channel Extraction** →  
4. **Signal Filtering** →  
5. **Frequency Analysis** →  
6. **Heart Rate Estimation**  

---

## 3. Assumptions and Dependencies
### Assumptions
- The subject's face is clearly visible.  
- Lighting conditions are stable.  

### Dependencies
- **Software**: OpenCV, NumPy, SciPy, Matplotlib  
- **Hardware**: Webcam or video input  

---

## 4. Inputs
### Parameters
- **Video File**: Video of a subject's face (e.g., `.mp4`)  
- **Window Size**: Number of frames for analysis (e.g., 50)  

### Source of Data
- Live webcam feed or video file  

---

## 5. Methodology/Architecture
### Detailed Design
1. **Face Detection**: Uses Haar Cascade to locate the face.  
2. **Green Channel Extraction**: Extracts average intensity from the green channel of the detected face region.  
3. **Signal Processing**: Applies a Butterworth bandpass filter to remove noise.  
4. **Frequency Analysis**: Uses FFT to find the dominant frequency within the heart rate range.  

### Algorithms Used
- Haar Cascade Classifier for face detection.  
- Butterworth filter for noise reduction.  
- FFT for frequency-domain heart rate estimation.  

---

## 6. Outputs
### Results
Estimated heart rate in beats per minute (BPM).  

### Format
Numerical value (e.g., "Heart Rate: 72 BPM").  

### Interpretation
Higher or lower values reflect physiological activity or potential issues.

---

## 7. Validation and Testing
### Validation Approach
Compared output against known heart rate benchmarks.  

### Testing Procedures
Tested on various pre-recorded videos with different lighting and movement.  

### Metrics Used
Mean absolute error (MAE) between estimated and ground truth heart rates.  

---

## 8. Usage Instructions
### Setup
Install required libraries:  
```bash
pip install opencv-python scipy matplotlib numpy
