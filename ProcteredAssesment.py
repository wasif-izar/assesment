import sys
import os
import base64
import torch
from flask import Flask, render_template, request, jsonify
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

# Judge0 API config
JUDGE0_URL = "https://judge0-ce.p.rapidapi.com"
HEADERS = {
    "X-RapidAPI-Key": "your_actual_rapidapi_key_here",  # Replace with your key
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
        "starter_code": "print('Hello, World!')"
    },
    {
        "id": 2,
        "title": "Sum of Two Numbers",
        "description": "Write a function that returns the sum of two numbers.",
        "starter_code": "def add(a, b):\n    return a + b"
    },
    {
        "id": 3,
        "title": "Check Palindrome",
        "description": "Check if a given string is a palindrome.",
        "starter_code": "def is_palindrome(s):\n    return s == s[::-1]"
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
        else:
            return jsonify({"error": "Timeout waiting for execution result"}), 408

        response_data = {
            "output": result_json.get("stdout", ""),
            "error": result_json.get("stderr", ""),
            "status": status,
            "time": result_json.get("time"),
            "memory": result_json.get("memory")
        }

        return jsonify(response_data)

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"API request failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/upload_webcam", methods=["POST"])
def upload_webcam():
    data = request.json
    image_data = data["image"].split(",")[1]
    image_bytes = base64.b64decode(image_data)

    if not os.path.exists("webcam_captures"):
        os.makedirs("webcam_captures")

    image_path = f"webcam_captures/capture_{len(os.listdir('webcam_captures'))}.png"
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    result = detect_suspicious_activity(image_path)
    return jsonify({"status": "success", "message": "Webcam image saved!", "suspicious": result})

def detect_suspicious_activity(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).float().to(device)
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)
    img /= 255.0

    with torch.no_grad():
        pred = model(img)[0]

    pred = non_max_suppression(pred, 0.4, 0.5)
    suspicious_classes = {67, 73, 77}  # Phone, laptop, book
    person_count = 0

    for det in pred:
        if det is not None and len(det):
            for *xyxy, conf, cls in det:
                class_id = int(cls.item())  
                if class_id == 0:
                    person_count += 1
                if class_id in suspicious_classes:
                    return True

    if person_count == 0 or person_count > 1:
        return True

    return False

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
