import torch
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import time

# Load the trained YOLOv8 model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

def predict_disease(image: Image.Image):
    start_time = time.time()  # Track inference time

    image_np = np.array(image.convert("RGB"))
    results = model.predict(image_np)

    detected_diseases = []
    detection_info = []

    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            confidence = box.conf[0].item()
            full_text = f"{label} ({confidence:.2f})"

            detected_diseases.append(label)
            detection_info.append({
                "Label": label,
                "Class Index": int(box.cls[0]),
                "Confidence": f"{confidence:.2f}",
                "Bounding Box": [x1, y1, x2, y2]
            })

            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 3)
            font_scale = max(0.5, (x2 - x1) / 550)
            thickness = max(1, int(font_scale * 3))
            text_size, _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_x, text_y = x1, max(y1 - 10, text_size[1] + 10)

            cv2.rectangle(image_np, (text_x, text_y - text_size[1] - 5), 
                          (text_x + text_size[0] + 10, text_y + 5), (0, 255, 0), -1)
            cv2.putText(image_np, full_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, (0, 0, 0), thickness)

    inference_time = round(time.time() - start_time, 2)  # Calculate inference time

    processed_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

    return processed_image, detected_diseases, inference_time, detection_info
