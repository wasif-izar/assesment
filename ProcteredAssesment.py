import sys
import os
import base64
import torch
from flask import Flask, render_template, request, jsonify, session
import cv2
import numpy as np
import requests
import time

# Add YOLOv5 to system path
sys.path.append("./yolov5")

# Import YOLOv5 utilities
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# Load YOLOv5 Model
device = select_device("")  # Selects GPU if available, else CPU
model = attempt_load("yolov5s.pt", device)

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Needed for session tracking

# Judge0 API config
JUDGE0_URL = "https://judge0-ce.p.rapidapi.com"
HEADERS = {
    "X-RapidAPI-Key": "d2cbc2bca7mshcfdb2fda562f0ffp1bd4a8jsnbf719a79c0b2",  # Replace with your real key
    "X-RapidAPI-Host": "judge0-ce.p.rapidapi.com",
    "Content-Type": "application/json"
}
MAX_RETRIES = 10
POLL_INTERVAL = 1

# Coding Questions
coding_questions = [
    {
        "id": 1,
        "title": "Print Hello World",
        "description": "Write a program that prints 'Hello, World!'",
        "starter_code": "print('Hello, World!')",
        "expected_output": "Hello, World!\n"
    },
    {
        "id": 2,
        "title": "Sum of Two Numbers",
        "description": "Write a function that returns the sum of two numbers.",
        "starter_code": "def add(a, b):\n    return a + b\n\nprint(add(2, 3))",
        "expected_output": "5\n"
    },
    {
        "id": 3,
        "title": "Check Palindrome",
        "description": "Check if a given string is a palindrome.",
        "starter_code": "def is_palindrome(s):\n    return s == s[::-1]\n\nprint(is_palindrome('madam'))",
        "expected_output": "True\n"
    }
]

@app.route("/")
def index():
    return render_template("question_selector.html", questions=coding_questions)

@app.route("/question/<int:question_id>")
def show_editor(question_id):
    question = next((q for q in coding_questions if q["id"] == question_id), None)
    if question is None:
        return "Question not found", 404
    return render_template("code_editor.html", question=question)  


@app.route("/submit_code", methods=["POST"])
def submit_code():
    try:
        data = request.json
        source_code = data.get("code", "")
        language_id = data.get("language_id", 71)
        stdin = data.get("stdin", "")
        question_id = int(data.get("question_id"))

        question = next((q for q in coding_questions if q["id"] == question_id), None)
        if not question:
            return jsonify({"error": "Invalid question ID"}), 400

        submission = {
            "language_id": language_id,
            "source_code": source_code,
            "stdin": stdin,
            "redirect_stderr_to_stdout": True
        }

        response = requests.post(
            f"{JUDGE0_URL}/submissions?base64_encoded=false&wait=false",
            json=submission,
            headers=HEADERS
        )
        response.raise_for_status()
        token = response.json()["token"]

        result_url = f"{JUDGE0_URL}/submissions/{token}?base64_encoded=false"
        for _ in range(MAX_RETRIES):
            result_response = requests.get(result_url, headers=HEADERS)
            result_response.raise_for_status()
            result_json = result_response.json()

            status = result_json["status"]["description"]
            if status not in ["In Queue", "Processing"]:
                break
            time.sleep(POLL_INTERVAL)

        output = result_json.get("stdout", "")
        expected_output = question["expected_output"]

        # Scoring logic
        score = 10 if output == expected_output else 0
        if "scores" not in session:
            session["scores"] = {}
        scores = session["scores"]
        scores[str(question_id)] = score
        session["scores"] = scores

        response_data = {
            "output": output,
            "error": result_json.get("stderr", ""),
            "status": status,
            "score": score,
            "time": result_json.get("time"),
            "memory": result_json.get("memory")
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/result")
def show_result():
    scores = session.get("scores", {})
    total_score = sum(scores.values())
    return f"<h2>Total Score: {total_score} / {len(coding_questions)*10}</h2><p>Per-question: {scores}</p>"

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

    # Run AI Model on Image
    result = detect_suspicious_activity(image_path)

    return jsonify({"status": "success", "message": "Webcam image saved!", "suspicious": result})


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
