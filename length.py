from flask import Flask, request, render_template_string, send_from_directory
import os
import cv2
import numpy as np
import math
from itertools import permutations
from typing import Tuple, List, Optional

# === Utility Functions ===

def get_extreme_points(cnt: np.ndarray) -> Tuple[Tuple[int, int], ...]:
    """
    Returns the extreme points (top, bottom, left, right) of a contour.
    """
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    return topmost, bottommost, leftmost, rightmost

def compute_distance_and_midpoint(p1: Tuple[int, int], 
                                  p2: Tuple[int, int], 
                                  p3: Tuple[int, int]) -> Tuple[float, Optional[Tuple[int, int]]]:
    """
    Calculates the perpendicular distance from p3 to the line segment formed by p1 and p2.
    Returns distance and the midpoint of p1 and p2 if segment is valid.
    """
    if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) < 500:
        return 0, None
    
    midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    distance = math.hypot(p3[0] - midpoint[0], p3[1] - midpoint[1])
    return distance, midpoint

def process_image(file_path: str, result_path: str) -> None:
    """
    Processes the uploaded image:
    - Finds contours
    - Determines extreme points
    - Calculates distances and draws results
    Saves processed image to result_path.
    """
    img = cv2.imread(file_path)
    output = img.copy()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 200 or h < 200:
            continue

        topmost, bottommost, leftmost, rightmost = get_extreme_points(cnt)
        extreme_points = [topmost, bottommost, leftmost, rightmost]

        max_distance = -1
        best_pts = ()
        best_midpoint = ()

        for p1, p2, p3 in permutations(extreme_points, 3):
            distance, midpoint = compute_distance_and_midpoint(p1, p2, p3)
            if distance > max_distance:
                max_distance = distance
                best_pts = (p1, p2, p3)
                best_midpoint = midpoint

        if best_pts and best_midpoint:
            _, _, p3 = best_pts
            cv2.line(output, p3, best_midpoint, (0, 255, 0), 2)
            label = f"{int(max_distance)} px"
            label_pos = (best_midpoint[0] + 10, best_midpoint[1] - 10)
            cv2.putText(output, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 225), 5)

    cv2.imwrite(result_path, output)

# === Flask App Setup ===

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# === Routes ===

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    """
    Handles file upload and image processing.
    """
    uploaded_file_url = None
    result_file_url = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No file part', 400
        file = request.files['image']
        if file.filename == '':
            return 'No selected file', 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        result_path = os.path.join(RESULT_FOLDER, 'result_' + file.filename)
        file.save(file_path)

        process_image(file_path, result_path)

        uploaded_file_url = f"/uploads/{file.filename}"
        result_file_url = f"/results/result_{file.filename}"

    html = """
    <h2>Upload an Image</h2>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*">
        <input type="submit" value="Upload">
    </form>

    {% if uploaded_file_url %}
        <h3>Original Image:</h3>
        <img src="{{ uploaded_file_url }}" width="300">
        <h3>Processed Image:</h3>
        <img src="{{ result_file_url }}" width="300">
    {% endif %}
    """
    return render_template_string(html, uploaded_file_url=uploaded_file_url, result_file_url=result_file_url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves uploaded file."""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results/<filename>')
def result_file(filename):
    """Serves result file."""
    return send_from_directory(RESULT_FOLDER, filename)

# === Run App ===

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)