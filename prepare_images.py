import cv2
import os
import json
from tqdm import tqdm
from consts import ANNOTATIONS_PATH, IMAGES_PATH, DataStructure, Record
from prepare_skeleton import remove_hand_and_draw_skeleton_blank, create_transition_sequence

def _crop_centered_box(image, bbox, image_size):
    output_size = (image_size, image_size)
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

def _process_custom_json(image_dir, json_path, gesture_label, output_dir, image_size):
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

        if os.path.exists(image_path) == False:
            continue

        image = cv2.imread(image_path)
        bboxes = info.bboxes

        for it, bbox in enumerate(bboxes):
            if ((len(bboxes) > 1 and info.labels[it] != gesture_label) or len(info.hand_landmarks[it]) == 0):
                continue
            
            crop = _crop_centered_box(image, bbox, image_size)

            filename = f"{image_id}.jpg"
            out_path = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, crop)

            # Create hands with mask and skeleton
            out_path = out_path.replace("cropped", "no_hands")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            out = remove_hand_and_draw_skeleton_blank(out_path, it, expand=30, fill_color=(255,255,255))
            cv2.imwrite(out_path, out)
            break

    print(f"✅ Cropped images saved to: {output_dir}")

def prepare_images(gestures, image_size):
    for gesture in gestures:
        for dirpath, _, filenames in os.walk(ANNOTATIONS_PATH):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if gesture not in file_path:
                    continue
                output_folder = os.path.join(dirpath.replace('annotations', 'cropped'), gesture)
                _process_custom_json(
                    image_dir=IMAGES_PATH,
                    json_path=file_path,
                    gesture_label=gesture,
                    output_dir=output_folder,
                    image_size=image_size
                )

# IMAGE_PATH = "./cropped/train/three_gun"
# for img_name in tqdm(os.listdir(IMAGE_PATH)):
#     path = os.path.join(IMAGE_PATH, img_name)
#     remove_hand_and_draw_skeleton_blank(path, expand=30, fill_color=(255,255,255)