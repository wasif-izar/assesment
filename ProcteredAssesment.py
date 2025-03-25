import sys
import os
import base64
import torch
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np

# 🔹 Add YOLOv5 to system path
sys.path.append("./yolov5")

# 🔹 Import YOLOv5 utilities
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# 🔹 Load YOLOv5 Model
device = select_device("")  # Selects GPU if available, else CPU
model = attempt_load("yolov5s.pt", device)

app = Flask(__name__)

questions = [
    {"question": "What is 2 + 2?", "options": ["3", "4", "5", "10"], "answer": "4"},
    {"question": "What is the capital of France?", "options": ["Berlin", "Paris", "Madrid"], "answer": "Paris"},
]

@app.route("/", methods=["GET", "POST"])
def quiz():
    if request.method == "POST":
        user_answers = request.form
        results = {}

        for i, q in enumerate(questions):
            user_answer = user_answers.get(f"q{i}", "No answer selected")
            is_correct = user_answer == q["answer"]
            results[f"q{i}"] = {"selected": user_answer, "correct": q["answer"], "is_correct": is_correct}

            # Print results in terminal
            print(f"Q{i+1}: {q['question']}")
            print(f"Your Answer: {user_answer}")
            print(f"Correct Answer: {q['answer']}")
            print(f"Result: {'✅ Correct' if is_correct else '❌ Incorrect'}")
            print("-" * 30)

        return render_template("submitted.html", results=results)

    return render_template("quiz.html", questions=questions, enumerate=enumerate)


# **Handle Webcam Image Upload & AI Detection**
@app.route("/upload_webcam", methods=["POST"])
def upload_webcam():
    data = request.json
    image_data = data["image"].split(",")[1]  # Remove the data URL header
    image_bytes = base64.b64decode(image_data)

    # Save image to disk
    if not os.path.exists("webcam_captures"):
        os.makedirs("webcam_captures")

    image_path = f"webcam_captures/capture_{len(os.listdir('webcam_captures'))}.png"
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    # **🔹 Run AI Model on Image**
    result = detect_suspicious_activity(image_path)

    return jsonify({"status": "success", "message": "Webcam image saved!", "suspicious": result})

# **AI Detection Function**
import cv2
import torch

def detect_suspicious_activity(image_path):
    img = cv2.imread(image_path)  # Load image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = torch.from_numpy(img).float().to(device)  # Convert to Tensor

    img = img.permute(2, 0, 1)  # Change shape from (H, W, C) to (C, H, W)
    img = img.unsqueeze(0)  # Add batch dimension (1, C, H, W)
    img /= 255.0  # Normalize to [0,1]

    with torch.no_grad():
        pred = model(img)[0]  # Get predictions

    pred = non_max_suppression(pred, 0.4, 0.5)

    suspicious_classes = {67, 73, 77}  # Phone, laptop, book
    person_count = 0

    for det in pred:
        if det is not None and len(det):
            for *xyxy, conf, cls in det:
                class_id = int(cls.item())  
                if class_id == 0:
                    person_count += 1  # Count people

                if class_id in suspicious_classes:
                    print(f"⚠️ Suspicious Object Detected: Class {class_id}")
                    return True

    if person_count == 0:
        print("⚠️ No person detected! Suspicious activity.")
        return True

    if person_count > 1:
        print("⚠️ Multiple people detected! Suspicious activity.")
        return True

    print("✅ Person detected. No suspicious activity.")
    return False

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
 