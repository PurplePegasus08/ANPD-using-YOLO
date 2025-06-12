# 🚗 Automatic Number Plate Detection (ANPD) using YOLO & PaddleOCR

Welcome to the **ANPD** project! This repository showcases a complete pipeline for **automatically detecting vehicle number plates** using **YOLOv8** and extracting the **text** from those plates using **PaddleOCR** — one of the most accurate open-source OCR engines.

---

## 📌 Overview

This project is designed to:
- Detect **license plates** in vehicle images using a YOLOv8 custom-trained model.
- **Crop** detected number plates from the original image.
- Use **PaddleOCR** to extract **text** from cropped plates.
- Save:
  - Cropped images
  - Annotated images with boxes
  - Final output text in a `.txt` file

✅ Everything runs locally with a single `.bat` file execution.

---

## 🧠 Technologies Used

| Component         | Technology               |
|------------------|--------------------------|
| Plate Detection  | YOLOv8 (custom-trained)  |
| OCR Engine       | PaddleOCR                |
| Language         | Python 3.10              |
| Libraries        | OpenCV, PaddleOCR, Ultralytics |

---

## 🗂️ Project Structure

```bash
ANPD/
├── main.py               # Main Python script
├── run_main.bat          # Batch file to run the script on Windows
├── best.pt               # YOLOv8 trained weights
├── requirements.txt      # Python dependencies (if needed)
└── main/
    ├── imagesand/        # 📥 Input images go here
    └── results/          # 📤 Output .txt, cropped & annotated images
