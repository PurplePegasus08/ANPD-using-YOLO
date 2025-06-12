# 🚗 Automatic Number Plate Detection (ANPD) using YOLO & PaddleOCR

Welcome to the **ANPD** project! This repository showcases a complete pipeline for **automatically detecting vehicle number plates** using YOLO (You Only Look Once) and **extracting the text** from those plates using **PaddleOCR** — one of the most accurate open-source OCR engines.

---

## 📌 Overview

This project is designed to:

* Detect **license plates** in vehicle images using an object detection model (YOLOv8 or similar).
* **Crop** detected number plates from the original image.
* Use **PaddleOCR** to extract the **alphanumeric text** from those cropped plate images.
* Output readable and structured results such as:
  `UP 81 BT 5486`

It’s fast, accurate, and modular — perfect for smart traffic systems, parking management, surveillance, and other real-world applications.

---

## 🧠 Technologies Used

| Component       | Technology              |
| --------------- | ----------------------- |
| Plate Detection | YOLOv8 (custom-trained) |
| OCR Engine      | PaddleOCR               |
| Language        | Python                  |
| Libraries       | OpenCV, PaddlePaddle    |

---

## ✅ GPU Setup Instructions (Windows)

To run PaddleOCR with GPU acceleration on Windows, follow these steps carefully:

---

### 🔧 1. Install CUDA Toolkit (11.2)

> 📌 PaddleOCR (with `paddlepaddle-gpu==2.6.0.post112`) is compatible with **CUDA 11.2**

#### ✅ Download and install:

* 🔗 [CUDA Toolkit 11.2 (Local Installer)](https://developer.nvidia.com/cuda-11.2.2-download-archive)

Install it to the default location:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2
```

---

### 📦 2. Install cuDNN for CUDA 11.2

> cuDNN is required by Paddle for GPU operations.

#### ✅ Download cuDNN 8.x for CUDA 11.2:

* 🔗 [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)

> Login → Choose **cuDNN for CUDA 11.2** → Download **cuDNN v8.1.1 (for CUDA 11.2)**

#### 📂 Copy files to your CUDA directory:

Extract and paste all files from `cudnn-windows-x64-v8.1.1.33` into:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2
```

That includes:

* `bin\cudnn64_8.dll` → into CUDA `bin`
* `include\cudnn*.h` → into CUDA `include`
* `lib\x64\cudnn*.lib` → into CUDA `lib\x64`

---

### 🛠️ 3. Set Environment Variables

Manually add the following paths to your **system environment variables** → `Path`:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp
```

> ✅ Restart your PC or command prompt after this step.

---

### 💾 4. Install Python Packages (for GPU)

Make sure you have Python 3.10 (same version used in the project).

#### ✅ Use these pip commands:

```bash
pip install paddlepaddle-gpu==2.6.0.post112 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
pip install paddleocr
pip install opencv-python
pip install ultralytics
```

> If you're using a **virtual environment**, activate it first.

---

### 🔍 5. Verify GPU is Detected

Run the following Python code to confirm Paddle is using GPU:

```python
import paddle
print("Is GPU available:", paddle.is_compiled_with_cuda())
```

Expected output:

```
Is GPU available: True
```

---

## ✅ Summary for GPU Setup

| Requirement      | Version       |
| ---------------- | ------------- |
| CUDA             | 11.2          |
| cuDNN            | 8.1.x         |
| PaddlePaddle GPU | 2.6.0.post112 |

If any errors occur (like `cudnn64_8.dll missing`), check:

* CUDA/cuDNN version match
* Environment paths
* PaddlePaddle version

---

## 📚 Folder Structure (Windows)

When run, the script automatically creates the following folder structure:

```
main/
├── imagesand/            # Put your input images here
├── results/              # Cropped plates, annotated images, and .txt results saved here
├── main.py               # Python script to run detection + OCR
├── run_main.bat          # Windows batch file for CLI usage
```

---

## ⚡ How to Use

### First Run (Folder Setup)

```
Double-click or run: run_main.bat
```

> ✅ It will create `imagesand` and `results` folders.

### Second Run (Detection + OCR)

```
run_main.bat image.jpg output.txt
```

> ✅ The script will:

* Detect number plates
* Crop and save images
* Extract text using PaddleOCR
* Save final text to `results/output.txt`

```
```
