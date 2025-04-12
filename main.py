import cv2
import torch
import time
import csv
import os
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import numpy as np

# Load the OWL-ViT model
print("Loading OWL-ViT model...")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Define initial custom classes
custom_classes = [
    "a lightbulb",
    "a monitor",
    "a lion",
    "a gaming console",
    "a matchstick"
]

# Logging setup
log_file = 'detections.csv'
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame_number", "class", "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"])

# Load video
video_path = "sample_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

fps_limit = 10
frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / fps_limit)
frame_number = 0

# FPS measurement
last_time = time.time()
frame_count = 0

print("🟢 Press 'e' to edit prompt classes. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_number % frame_interval != 0:
        frame_number += 1
        continue

    # Convert to PIL image
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # Preprocess
    inputs = processor(text=custom_classes, images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([pil_image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.2)[0]

    # Draw detections
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        confidence = float(score)
        label = label.item()
        label_text = custom_classes[label]
        box = [int(x) for x in box.tolist()]

        if confidence >= 0.2:
            # Rectangle
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            # Label with background
            label_str = f"{label_text} ({confidence*100:.1f}%)"
            (w, h), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(frame, (box[0], box[1] - h - 10), (box[0] + w, box[1]), (0, 255, 0), -1)
            cv2.putText(frame, label_str, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

            # Log
            with open(log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([frame_number, label_text, confidence, *box])

    # FPS counter
    frame_count += 1
    elapsed = time.time() - last_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        last_time = time.time()
        frame_count = 0
    else:
        fps = 0

    # Display FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 50, 50), 2)

    # Resize and display
    display_frame = cv2.resize(frame, (960, 720))
    cv2.imshow("Zero-Shot Detection", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("e"):
        print("✏️ Enter new comma-separated prompts:")
        new_input = input("📝 New classes: ").strip()
        if new_input:
            custom_classes = [c.strip() for c in new_input.split(",")]
            print("✅ Updated classes:", custom_classes)

    frame_number += 1

cap.release()
cv2.destroyAllWindows()
