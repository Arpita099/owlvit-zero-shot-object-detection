# OWL-ViT Zero-Shot Object Detection

This project implements **zero-shot object detection** using Google's [OWL-ViT](https://huggingface.co/google/owlvit-base-patch32) model. It allows detecting **custom object classes** from video input without retraining the model — just by using **text prompts**!

---

## Features

-  **Zero-shot detection** with custom classes (e.g., "a lightbulb", "a gaming console")
- Works with any video file (e.g., `.mp4`)
- **Live class editing** (press `e` to update prompts during runtime)
- Optimized to run at ~10 FPS
- **Prediction logs** saved to `detections.csv`
- Clean and ready for ONNX or TorchScript conversion

---

##  Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Arpita099/owlvit-zero-shot-object-detection.git
cd owlvit-zero-shot-object-detection
