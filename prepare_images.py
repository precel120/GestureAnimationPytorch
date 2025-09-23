import cv2
import numpy as np
from dataclasses import dataclass
import mediapipe as mp
import os
from typing import List, Dict
import json
from tqdm import tqdm
from consts import IMAGE_SIZE, GESTURES, ANNOTATIONS_PATH, IMAGES_PATH, CROPPED_PATH

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

def remove_hand_and_draw_skeleton_blank(path, expand=60, fill_color=(255,255,255)):
    image_bgr = cv2.imread(path)
    h, w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks or len(result.multi_hand_landmarks) <= 3:
        return

    # --- STEP 1: create mask ---
    mask = np.zeros((h, w), dtype=np.uint8)
    for hand in result.multi_hand_landmarks:
        pts = np.array([(int(lm.x * w), int(lm.y * h)) for lm in hand.landmark], dtype=np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)

    # expand mask so it covers entire hand
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand, expand))
    mask = cv2.dilate(mask, kernel, iterations=1)

    # --- STEP 2: remove hand by filling ---
    out = image_bgr.copy()
    out[mask == 255] = fill_color  # paint hand area white (or black)

    # --- STEP 3: draw skeleton ---
    for hand in result.multi_hand_landmarks:
        mp_drawing.draw_landmarks(out, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imwrite(path.replace("cropped", "temp"), out)

def replace_gesture(source_gesture, target_gesture, expand=60, fill_color=(255,255,255)):
    image_bgr = cv2.imread(target_gesture)
    h, w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    result = hands.process(img_rgb)

    # --- STEP 1: create mask ---
    mask = np.zeros((h, w), dtype=np.uint8)
    for hand in result.multi_hand_landmarks:
        pts = np.array([(int(lm.x * w), int(lm.y * h)) for lm in hand.landmark], dtype=np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)

    # expand mask so it covers entire hand
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand, expand))
    mask = cv2.dilate(mask, kernel, iterations=1)

    # --- STEP 2: remove hand by filling ---
    out = cv2.imread(source_gesture)
    out[mask == 255] = fill_color  # paint hand area white (or black)

    # --- STEP 3: draw skeleton ---
    for hand in result.multi_hand_landmarks:
        mp_drawing.draw_landmarks(out, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imwrite('./replace.jpg', out)

# replace_gesture('./0a9ddce1-2cd4-4b14-9e43-3889b6a6e87c.jpg', './0a14f51b-4cf4-4694-adb2-4f3f41ae9f19.jpg', 40)

def _crop_centered_box(image, bbox, output_size=(IMAGE_SIZE, IMAGE_SIZE)):
    img_h, img_w = image.shape[:2]
    x, y, w, h = bbox

    # Convert from normalized to absolute coords
    x = int(x * img_w)
    y = int(y * img_h)
    w = int(w * img_w)
    h = int(h * img_h)

    center_x = x + w // 2
    center_y = y + h // 2

    crop_w, crop_h = output_size
    left = center_x - crop_w // 2
    top = center_y - crop_h // 2
    right = left + crop_w
    bottom = top + crop_h

    # Ensure crop is within image bounds
    if left < 0:
        left = 0
        right = crop_w
    if top < 0:
        top = 0
        bottom = crop_h
    if right > img_w:
        right = img_w
        left = img_w - crop_w
    if bottom > img_h:
        bottom = img_h
        top = img_h - crop_h

    left = max(0, left)
    top = max(0, top)
    right = min(img_w, right)
    bottom = min(img_h, bottom)

    crop = image[top:bottom, left:right]
    return crop

@dataclass
class Record:
    bboxes: List[List[float]]  # List of [x, y, width, height]
    labels: List[str]
    hand_landmarks: List[List[List[float]]]  # 2D landmarks per hand
DataStructure = Dict[str, Record]

def _process_custom_json(image_dir, json_path, gesture_label, output_dir, output_size=(IMAGE_SIZE, IMAGE_SIZE)):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        temp = json.load(f)
        data: DataStructure = {
            k: Record(
                bboxes=v["bboxes"],
                labels=v["labels"],
                hand_landmarks=v["hand_landmarks"],
            )
            for k, v in temp.items()
    }

    for image_id, info in tqdm(data.items()):
        image_path = os.path.join(image_dir, f"{gesture_label}/{image_id}.jpg")
        os.makedirs(os.path.join(image_dir, gesture_label), exist_ok=True)

        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Cannot read image: {image_path}")
            continue

        bboxes = info.bboxes

        for it, bbox in enumerate(bboxes):
            if ((len(bboxes) > 1 and info.labels[it] != gesture_label) or len(info.hand_landmarks[it]) == 0):
                continue
            crop = _crop_centered_box(image, bbox, output_size)

            filename = f"{image_id}.jpg"
            out_path = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, crop)

    print(f"✅ Cropped images saved to: {output_dir}")

def prepare_images():
    for gesture in GESTURES:
        for dirpath, _, filenames in os.walk(ANNOTATIONS_PATH):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                output_folder = os.path.join(dirpath.replace('annotations', 'cropped'), gesture)
                if gesture not in file_path:
                    continue
                _process_custom_json(
                    image_dir=IMAGES_PATH,
                    json_path=file_path,
                    gesture_label=gesture,
                    output_dir=output_folder,
                    output_size=(IMAGE_SIZE, IMAGE_SIZE)
                )

# prepare_images()

IMAGE_PATH = "./cropped/train/three_gun"
for img_name in tqdm(os.listdir(IMAGE_PATH)):
    path = os.path.join(IMAGE_PATH, img_name)
    remove_hand_and_draw_skeleton_blank(path, expand=20, fill_color=(255,255,255))
