# 🚗 Automatic Number Plate Detection (ANPD) using YOLO & PaddleOCR

Welcome to the **ANPD** project! This repository showcases a complete pipeline for **automatically detecting vehicle number plates** using YOLO (You Only Look Once) and **extracting the text** from those plates using **PaddleOCR** — one of the most accurate open-source OCR engines.

---

## 📌 Overview

This project is designed to:
- Detect **license plates** in vehicle images using an object detection model (YOLOv8 or similar).
- **Crop** detected number plates from the original image.
- Use **PaddleOCR** to extract the **alphanumeric text** from those cropped plate images.
- Output readable and structured results such as:  
  `UP 81 BT 5486`

It’s fast, accurate, and modular — perfect for smart traffic systems, parking management, surveillance, and other real-world applications.

---

## 🧠 Technologies Used

| Component         | Technology               |
|------------------|--------------------------|
| Plate Detection  | YOLOv8 (custom-trained)  |
| OCR Engine       | PaddleOCR                |
| Language         | Python                   |
| Libraries        | OpenCV, PaddlePaddle     |

---



