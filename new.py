import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import permutations
from typing import Tuple, Optional

# === Utility Functions ===

def get_extreme_points(cnt: np.ndarray) -> Tuple[Tuple[int, int], ...]:
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    return topmost, bottommost, leftmost, rightmost

def compute_distance_and_midpoint(p1: Tuple[int, int], 
                                  p2: Tuple[int, int], 
                                  p3: Tuple[int, int]) -> Tuple[float, Optional[Tuple[int, int]]]:
    if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) < 500:
        return 0, None
    midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    distance = math.hypot(p3[0] - midpoint[0], p3[1] - midpoint[1])
    return distance, midpoint

def process_image(img: np.ndarray) -> np.ndarray:
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

    return output

# === Main ===

# Replace with your image path
image_path = r"C:\Users\sasmitha\OneDrive\Desktop\tack_object\color3.bmp"

img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

result_img = process_image(img)

# Convert BGR to RGB for matplotlib

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

# === Show original and result side by side ===
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(result_rgb)
plt.title("Processed Image")
plt.axis('off')

plt.tight_layout()
plt.show()
