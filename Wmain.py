import os
import sys
import cv2
from ultralytics import YOLO

# 🧠 Force PaddleOCR to run on CPU (no cudnn errors)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from paddleocr import PaddleOCR
import paddle
paddle.set_device("cpu")

# ==== Create folders ====
base_dir = 'main'
image_dir = os.path.join(base_dir, 'imagesand')
result_dir = os.path.join(base_dir, 'results')

os.makedirs(image_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# ==== First run just creates folders ====
if len(sys.argv) < 3:
    print("📁 Folders created (if not already existing):")
    print(f"🖼️  Place your image in: {image_dir}")
    print("🚀 Re-run this script like:")
    print("    run_main.bat <image_name> <output_text_file_name>")
    print("Example:")
    print("    run_main.bat car1.jpg output1.txt")
    sys.exit(0)

# ==== Input filenames ====
image_name = sys.argv[1]
output_text_name = sys.argv[2]

if not output_text_name.lower().endswith('.txt'):
    output_text_name += '.txt'

image_path = os.path.join(image_dir, image_name)
output_text_path = os.path.join(result_dir, output_text_name)

if not os.path.exists(image_path):
    print(f"❌ Image not found: {image_path}")
    sys.exit(1)

# ==== Load models ====
model = YOLO("best3.pt")  # Adjust path to your model
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

# ==== Run YOLO ====
results = model.predict(image_path, imgsz=320, conf=0.2)
detected_plates = []

for i, r in enumerate(results):
    boxes = r.boxes.xyxy.cpu().numpy().astype(int)
    orig_img = cv2.imread(image_path)

    for j, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        crop = orig_img[y1:y2, x1:x2]

        crop_name = f"crop_{i}_{j}.jpg"
        crop_path = os.path.join(result_dir, crop_name)
        cv2.imwrite(crop_path, crop)

        # ==== OCR on cropped image ====
        result = ocr.ocr(crop_path, cls=True)
        plate_text = []

        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                cleaned_text = ''.join(c for c in text if c.isalnum())
                plate_text.append(cleaned_text)

        final_text = ''.join(plate_text) if plate_text else "NoTextDetected"
        detected_plates.append(final_text)

        # ==== Draw box ====
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save annotated image
    annotated_path = os.path.join(result_dir, f"annotated_{image_name}")
    cv2.imwrite(annotated_path, orig_img)

# ==== Save text to .txt ====
with open(output_text_path, 'w', encoding='utf-8') as f:
    for plate in detected_plates:
        f.write(plate + '\n')  # one plate per line

# ==== Done ====
print(f"\n✅ Detected plate(s): {', '.join(detected_plates)}")
print(f"📄 Text saved to: {output_text_path}")
print(f"📸 Cropped and annotated images saved in: {result_dir}")
