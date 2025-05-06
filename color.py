import cv2
import os
import numpy as np

def white_balance_grayworld(img):
    result = img.copy().astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3

    result[:, :, 0] *= avg_gray / avg_b
    result[:, :, 1] *= avg_gray / avg_g
    result[:, :, 2] *= avg_gray / avg_r

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

# Paths
input_folder = r"C:\Users\sasmitha\OneDrive\Desktop\tack_object"
output_folder = r"C:\Users\sasmitha\OneDrive\Desktop\output"
os.makedirs(output_folder, exist_ok=True)

ref_a = 139.6
ref_b = 145.1

# Process each image
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        continue

    img_path = os.path.join(input_folder, filename)
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
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 200 or h < 200:
            continue

        # Create mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        wb_img = white_balance_grayworld(img)
        masked_img = cv2.bitwise_and(wb_img, wb_img, mask=mask)

        lab = cv2.cvtColor(masked_img, cv2.COLOR_BGR2LAB)
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]

        mean_a = cv2.mean(a_channel, mask=mask)[0]
        mean_b = cv2.mean(b_channel, mask=mask)[0]

        color_dist = np.sqrt((mean_a - ref_a) ** 2 + (mean_b - ref_b) ** 2)
        print(f"[{filename}] mean_a: {mean_a:.2f}, mean_b: {mean_b:.2f}, color_dist: {color_dist:.2f}")

        if color_dist > 36:
            box_color = (0, 255, 0)
            label = "OK"
        else:
            box_color = (0, 0, 255)
            label = "Not OK"

        cv2.rectangle(output, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # Save output image per object (optional)
        # crop_output_path = os.path.join(output_folder, f'{label}_{os.path.splitext(filename)[0]}_{i}.png')
        # cv2.imwrite(crop_output_path, wb_img[y:y+h, x:x+w])

    # Save final output image with boxes
    output_image_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_boxed.png')
    cv2.imwrite(output_image_path, output)

print("Done.")
