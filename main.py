import os
import sys
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ==== Create main folders ====
base_dir = 'main'
image_dir = os.path.join(base_dir, 'imagesand')
result_dir = os.path.join(base_dir, 'results')

os.makedirs(image_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# ==== Check if this is first run (folders just created) ====
if len(sys.argv) < 3:
    print("📁 Folders created (if not already existing):")
    print(f"🖼️  Place your image in: {image_dir}")
    print("🚀 Re-run this script like:")
    print("    python3 main.py <image_name> <output_text_file_name>")
    print("Example:")
    print("    python3 main.py car1.jpg output1.txt")
    sys.exit(0)

# ==== Get image and txt name from arguments ====
image_name = sys.argv[1]
output_text_name = sys.argv[2]

# ==== Ensure txt name ends with .txt ====
if not output_text_name.lower().endswith('.txt'):
    output_text_name += '.txt'

# ==== Full paths ====
image_path = os.path.join(image_dir, image_name)
output_text_path = os.path.join(result_dir, output_text_name)

# ==== Load models ====
model = YOLO("/home/pegasus/Documents/NUMBERplates/best3.pt")
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# ==== Run YOLO detection ====
results = model.predict(image_path, imgsz=320, conf=0.2)
detected_plates = []

for i, r in enumerate(results):
    boxes = r.boxes.xyxy.cpu().numpy().astype(int)
    orig_img = cv2.imread(image_path)

    for j, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        crop = orig_img[y1:y2, x1:x2]

        # ==== Save cropped image ====
        crop_name = f"crop_{i}_{j}.jpg"
        crop_path = os.path.join(result_dir, crop_name)
        cv2.imwrite(crop_path, crop)

        # ==== OCR ====
        result = ocr.ocr(crop_path, cls=True)
        plate_text = []

        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                cleaned_text = ''.join(c for c in text if c.isalnum())  # Remove special characters
                plate_text.append(cleaned_text)

        final_text = ''.join(plate_text) if plate_text else "NoTextDetected"
        detected_plates.append(final_text)

        # ==== Draw bounding box ====
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ==== Save annotated image ====
    annotated_path = os.path.join(result_dir, f"annotated_{image_name}")
    cv2.imwrite(annotated_path, orig_img)

# ==== Combine and save result ====
joined_plate_line = ' '.join(detected_plates)

with open(output_text_path, 'w') as f:
    f.write(joined_plate_line + '\n')

# ==== Print to terminal ====
print(f"\n✅ Detected plate(s): {joined_plate_line}")
print(f"📄 Text saved to: {output_text_path}")
print(f"📸 Cropped and annotated images saved in: {result_dir}")
