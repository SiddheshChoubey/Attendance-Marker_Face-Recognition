# 📸 Smart Attendance System using Face Detection

### 👨‍💻 A Software Engineering Mini Project

This project is a **Face Recognition-Based Attendance System** built using **OpenCV**, **Haar Cascade**, and the **face_recognition** library in Python. It captures live video from a webcam, detects and recognizes faces, and logs attendance automatically.

---

## 🧰 Libraries Used

- **NumPy**  
  Supports multi-dimensional arrays and a collection of mathematical functions to operate on them.

- **Pandas**  
  Powerful and easy-to-use data analysis and manipulation tool built on Python.

- **OpenCV**  
  Real-time computer vision library used for image processing and video capture.

- **Haar Cascade**  
  Machine learning-based object detection algorithm used for identifying faces in an image or video.

- **face_recognition**  
  Simplest and most powerful face recognition library for Python.

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/smart-attendance-system.git
cd smart-attendance-system
```

### 2️⃣ Set Up a Virtual Environment
```bash
python -m venv env
```

### 3️⃣ Activate the Virtual Environment

- **Windows:**
  ```bash
  .\env\Scripts\activate
  ```

- **macOS/Linux:**
  ```bash
  source env/bin/activate
  ```

### 4️⃣ Install Required Packages
```bash
pip install opencv-contrib-python numpy pandas Pillow pytest-shutil python-csv face_recognition
```

### 5️⃣ Run the Project
```bash
python main.py
```

---

## 📌 Notes

- Make sure your webcam is enabled.
- Images of known faces must be placed in the `known_faces/` directory, each image named after the person it represents.
- Attendance will be logged in `attendancelog.csv`.

---
