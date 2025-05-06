import math
import os

import cv2
import numpy as np


def get_extreme_points(cnt):
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    return topmost, bottommost, leftmost, rightmost


def compute_distance_and_midpoint(p1, p2, p3):
    if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) < 500:
        return 0, None

    midpoint_x = (p1[0] + p2[0]) // 2
    midpoint_y = (p1[1] + p2[1]) // 2
    midpoint = (midpoint_x, midpoint_y)

    distance = math.hypot(p3[0] - midpoint_x, p3[1] - midpoint_y)
    return distance, midpoint


# Constants
INPUT_FOLDER = r"C:\Users\sasmitha\OneDrive\Desktop\tack_object"
OUTPUT_FOLDER = r"C:\Users\sasmitha\OneDrive\Desktop\output_dis"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Process each image
for filename in os.listdir(INPUT_FOLDER):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        continue

    img_path = os.path.join(INPUT_FOLDER, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Skipping {filename} (could not load).")
        continue

    output = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 200 or h < 200:
            continue

        topmost, bottommost, leftmost, rightmost = get_extreme_points(cnt)
        max_distance = 0
        pt1 = pt2 = pt3 = midpoint = None

        extreme_points = [topmost, bottommost, leftmost, rightmost]

        for pt in extreme_points:
            cv2.circle(output, pt, 15, (0, 0, 255), -1)

        for i in extreme_points:
            for j in [x for x in extreme_points if x != i]:
                for k in [x for x in extreme_points if x != i and x != j]:
                    distance, mid = compute_distance_and_midpoint(i, j, k)
                    if distance > max_distance:
                        max_distance = distance
                        pt1, pt2, pt3, midpoint = i, j, k, mid

        if pt1 and pt2 and pt3 and midpoint:
            cv2.line(output, pt3, midpoint, (0, 255, 0), 2)
            cv2.circle(output, pt3, 5, (255, 255, 0), -1)
            label_pos = ((pt3[0] + midpoint[0]) // 2, (pt3[1] + midpoint[1]) // 2)
            cv2.putText(
                output,
                f"{max_distance:.1f}",
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        output_image_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}_boxed.png")
        cv2.imwrite(output_image_path, output)

print("Done.")
