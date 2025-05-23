import os
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
from datetime import datetime

# ==== Ask user for inputs ====
image_folder = 'your_images_folder'  # Replace with your folder path or keep it same as script
image_name = input("🖼️ Enter image name (e.g., 2.png): ").strip()
output_text_name = input("📄 Enter output text file name (e.g., result.txt): ").strip()

# ==== Set up paths ====
image_path = os.path.join(image_folder, image_name)
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
output_text_path = os.path.join(output_dir, output_text_name)

# ==== Load Model ====
model = YOLO("/home/pegasus/Documents/NUMBERplates/best.pt")
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# ==== Run detection ====
results = model.predict(image_path, imgsz=320, conf=0.2)

cropped_texts = []

for i, r in enumerate(results):
    boxes = r.boxes.xyxy.cpu().numpy().astype(int)
    orig_img = cv2.imread(image_path)

    for j, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        crop = orig_img[y1:y2, x1:x2]

        # Save cropped image
        crop_name = f"crop_{i}_{j}.jpg"
        crop_path = os.path.join(output_dir, crop_name)
        cv2.imwrite(crop_path, crop)

        # OCR
        result = ocr.ocr(crop_path, cls=True)
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                cropped_texts.append(text)
        else:
            cropped_texts.append("No text detected")

        # Draw bounding box
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save annotated image
    annotated_path = os.path.join(output_dir, f"annotated_{image_name}")
    cv2.imwrite(annotated_path, orig_img)

# ==== Save all extracted text ====
with open(output_text_path, 'w') as f:
    for idx, t in enumerate(cropped_texts):
        f.write(f"Plate {idx+1}: {t}\n")

print(f"\n✅ Text saved to: {output_text_path}")
print(f"📸 Cropped images and annotated image saved in: {output_dir}")
